from typing import Optional, Sequence, Tuple, Union, NamedTuple

import jax
import jax.numpy as jnp
import chex
from chex import PRNGKey, Array, dataclass # standard dataclass doesnt work with jax since its not a valid jax type, they seem to use it in jumanjki, dont know how it works, maybe it got explicitly registered but dont know where they do that
from jraph import GraphsTuple

from jumanji.environments.routing.tsp.types import Observation as TspObservation
from jumanji.environments.routing.tsp.types import State as TspState
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.environments import TSP
from jumanji.environments.routing.tsp.reward import RewardFn, distance_between_two_cities
from jumanji.environments.routing.tsp.generator import Generator, UniformGenerator
# from jumanji.environments.routing.tsp.reward import RewardFn, distance_between_two_cities
from moco.tsp_actors import knn_graph
import jraph

class SparseTspReward(RewardFn):
    """Small modification of the reward function from jumanji from -num_cities*sqrt(2) to -num_unvisited_cities*sqrt(2)"""

    def __call__(
        self, state: TspState, action: chex.Array, next_state: TspState, is_valid: bool
    ) -> chex.Numeric:
        num_cities = len(state.visited_mask)
        previous_city = state.coordinates[state.position]
        next_city = next_state.coordinates[next_state.position]
        num_remaining = num_cities - state.num_visited
        # By default, returns the negative distance between the previous and new city.
        reward = jax.lax.select(
            is_valid,
            -distance_between_two_cities(previous_city, next_city),
            jnp.array(-num_remaining * jnp.sqrt(2), float),
        )
        # Returns 0 for the first city selected.
        reward = jax.lax.select(
            state.num_visited == 0,
            jnp.array(0, float),
            reward,
        )
        # Adds the distance between the last city and the first city if the tour is finished.
        initial_city = state.coordinates[state.trajectory[0]]
        reward = jax.lax.select(
            jnp.all(next_state.visited_mask),
            reward - distance_between_two_cities(next_city, initial_city),
            reward,
        )
        return reward

@dataclass
class GraphTspState(TspState):
    """adds a graph to the state to enable knn graph to be added to the observation"""
    graph: GraphsTuple

class GraphTspObservation(NamedTuple):
    """adds a graph to the observation to enable knn graph to be added to the observation"""
    # subclassing from the original TspObservation didnt work since it is a NamedTuple so I just copied the structure and added a field
    # probably would be better if it were a dataclass like state but I want to keep it consistent with the original
    coordinates: chex.Array  # (num_cities, 2)
    position: chex.Numeric  # ()
    trajectory: chex.Array  # (num_cities,)
    action_mask: chex.Array  # (num_cities,)
    graph: GraphsTuple

class CustomTSP(TSP):
    def __init__(self, reward_fn: RewardFn = None, **kwargs, ):
        super().__init__(**kwargs)
        self.reward_fn = reward_fn or SparseTspReward()

    def reset_from_problem(self, coordinates: Array) -> Tuple[TspState, TimeStep]:
        """Resets the environment.

        Args:
            coordinates: The coordinates of the cities.

        Returns:
            The initial state and timestep.
        """
        state = TspState(
            coordinates=coordinates,
            position=jnp.array(-1, jnp.int32),
            visited_mask=jnp.zeros(self.num_cities, dtype=bool),
            trajectory=jnp.full(self.num_cities, -1, jnp.int32),
            num_visited=jnp.array(0, jnp.int32),
            key=None,
        )
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep
    
    def make_dummy_state(self) -> TspState:
        """Creates a dummy state for the environment.

        Args:
            key: A random key.

        Returns:
            A dummy state.
        """
        state = TspState(
            coordinates=jnp.zeros((self.num_cities, 2)),
            position=jnp.array(-1, jnp.int32),
            visited_mask=jnp.zeros(self.num_cities, dtype=bool),
            trajectory=jnp.full(self.num_cities, -1, jnp.int32),
            num_visited=jnp.array(0, jnp.int32),
            key=jax.random.PRNGKey(0),
        )
        return state
    
class CustomGraphTSP(TSP):
    """TSP environment which adds a knn graph to the observation."""

    def __init__(self, k: int, reward_fn: RewardFn = None, **kwargs):
        super().__init__(**kwargs)
        self.reward_fn = reward_fn or SparseTspReward()
        self.k = k

    # a bit hacky since step defines the action to be the node next to be inserted given by the node index
    # probably would be much cleaner for the sparsified graph env if it accepted the next edge to be inserted
    # as the action given by the edge index. this would be much cleaner for the actor 
    # since it can just do a softmax over the edges and if needed mask out the ones not connected to the current node
    # and then just take the argmax of the resulting distribution to get the next edge to be inserted (or sample)
    def step(
        self, state: GraphTspState, action: chex.Numeric
    ) -> Tuple[GraphTspState, TimeStep[GraphTspObservation]]:
        """Modified from jumanji TSP step function to mark the episode as done if the action leads to a state 
        with no feasible actions left. This can happen with a k-NN graph if the action leads to a city that has
        no feasible actions left. This would lead to the next state with all logits being -inf 
        since we mask out infeasible actions which makes the pgl -inf. We thus do not want to include
        this step in the pg loss and terminate the episode in this step.
        We also have to emit the penalty for resulting in an infeasible tour already in this step. Thus the condition
        passed to the reward function is the conjunction of the is_valid condition and the condition that the next state
        has feasible actions left.
        """
        first_city_selected = state.num_visited == 0
        out_edges = state.graph.receivers.reshape(self.num_cities, self.k)[state.position]
        edge_in_graph = jnp.any(action == out_edges)
        already_visited = state.visited_mask[action]
        is_valid = first_city_selected | (edge_in_graph & ~already_visited)
        
        next_state = jax.lax.cond(
            is_valid,
            self._update_state,
            lambda *_: state,
            state,
            action,
        )
        
        # all actions are infeasible by default
        next_action_feasible = jnp.full((self.num_cities), False)
        # set the cities that can be reached from the next_states position to feasible
        next_action_feasible = next_action_feasible.at[next_state.graph.receivers.reshape(self.num_cities, self.k)[next_state.position]].set(True)
        # set the cities that have already been visited to infeasible
        next_action_feasible = jnp.where(next_state.visited_mask == 1, False, next_action_feasible)
        number_of_feasible_actions_next_state = jnp.sum(next_action_feasible)
        # if there are no feasible actions left or the action is invalid but the tour is not yet complete, penalize the agent
        next_state_feasible = ~((number_of_feasible_actions_next_state == 0) & ~(next_state.num_visited == self.num_cities))
        reward = self.reward_fn(state, action, next_state,  is_valid & next_state_feasible)
        observation = self._state_to_observation(next_state)

        # Terminate if all cities have been visited or the action is invalid or the next state has no feasible actions left
        is_done = (next_state.num_visited == self.num_cities) | ~is_valid | (number_of_feasible_actions_next_state == 0)
        timestep = jax.lax.cond(
            is_done,
            termination,
            transition,
            reward,
            observation,
        )
        return next_state, timestep

    def reset_from_problem(self, coordinates: Array) -> Tuple[GraphTspState, TimeStep]:
        """Resets the environment from a given problem description by coordinates.
        Args:
            coordinates: The coordinates of the cities.
        Returns:
            The initial state and timestep.
        """
        graph = knn_graph(coordinates, self.k)
        state = GraphTspState(
            coordinates=coordinates,
            position=jnp.array(-1, jnp.int32),
            visited_mask=jnp.zeros(self.num_cities, dtype=bool),
            trajectory=jnp.full(self.num_cities, -1, jnp.int32),
            num_visited=jnp.array(0, jnp.int32),
            key=None,
            graph=graph
        )
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep
    
    def make_dummy_state(self) -> TspState:
        """Creates a dummy state for the environment.
        Returns:
            A dummy state.
        """
        key = jax.random.PRNGKey(0)
        coordinates = jax.random.uniform(key, (self.num_cities, 2))
        graph = knn_graph(coordinates, self.k)
        state = GraphTspState(
            coordinates=coordinates,
            position=jnp.array(-1, jnp.int32),
            visited_mask=jnp.zeros(self.num_cities, dtype=bool),
            trajectory=jnp.full(self.num_cities, -1, jnp.int32),
            num_visited=jnp.array(0, jnp.int32),
            key=key,
            graph=graph
        )
        return state

    def __repr__(self) -> str:
        return f"TSP environment with {self.num_cities} cities sparsified to {self.k} neighbors."

    def reset(self, key: PRNGKey) -> Tuple[GraphTspState, TimeStep[GraphTspObservation]]:
        """Resets the environment."""
        state = self.generator(key)
        graph = knn_graph(state.coordinates, self.k)
        state = GraphTspState(**state, graph=graph)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def _update_state(self, state: GraphTspState, action: chex.Numeric) -> GraphTspState:
        _state = super()._update_state(state, action)
        return GraphTspState(**_state, graph=state.graph)
    
    def _state_to_observation(self, state: GraphTspState) -> GraphTspObservation:
        observation = super()._state_to_observation(state)
        # return GraphTspObservation(**observation, graph=state.graph) 
        # observation is a NamedTuple which doesnt support ** unpacking since it is not a mapping in contrst to state which is a dataclass
        # so we have to do it manually
        return GraphTspObservation(
            coordinates=observation.coordinates,
            position=observation.position,
            trajectory=observation.trajectory,
            action_mask=observation.action_mask,
            graph=state.graph
        )
    
@dataclass
class MisState:
    problem: jraph.GraphsTuple
    assignment: Array
    is_done: Array
    adjacency_matrix: Array
    graph_masks: Array # 1 where node belongs to graph
    graph_ids: Array

@dataclass
class MisObservation:
    problem: jraph.GraphsTuple
    assignment: Array
    action_masks: Array # 1 where action is invalid

def action_infeasibility(state, action):
    """Returns True if action is infeasible, False otherwise"""
    return jnp.logical_and(state.adjacency_matrix[action], state.assignment).any()

class MaxIndependentSet:
    def reset_from_problem(self, problem):
        num_total_nodes = problem.nodes.shape[0]
        num_graphs = problem.n_node.shape[0]
        init_assigment = jnp.zeros(num_total_nodes, dtype=jnp.bool_)
        is_done = jnp.zeros(num_graphs, dtype=jnp.bool_)
        adjacency = jnp.zeros((num_total_nodes,num_total_nodes), dtype=jnp.bool_)
        adjacency = adjacency.at[problem.senders, problem.receivers].set(1)

        # create graph masks
        end_ind = problem.n_node.cumsum()
        start_ind = jnp.roll(end_ind, 1).at[0].set(0)
        node_indices = jnp.arange(num_total_nodes)
        graph_masks = jnp.logical_and(node_indices >= start_ind[:, None], node_indices < end_ind[:, None])
        graph_ids = jnp.repeat(jnp.arange(problem.n_node.shape[0]), problem.n_node)

        state = MisState(problem=problem, assignment=init_assigment, is_done=is_done, adjacency_matrix=adjacency, graph_masks=graph_masks, graph_ids=graph_ids)
        obs = MisObservation(problem=problem, assignment=init_assigment, action_masks=~graph_masks)
        return state, obs

    def step(self, state, action):
        # check if action is valid: check if already part of assignment and if insertion would violate adjacency
        is_infeasible = jax.vmap(action_infeasibility, in_axes=(None, 0))(state, action)
        is_already_inserted = state.assignment[action]
        is_valid = ~jnp.logical_or(is_infeasible, is_already_inserted)

        # update assignment for valid actions
        # should be set to True if is valid or if it is invalid but already True in the assignment
        # secondary can happen when only invalid actions are the left due to the episode being is done but the actor still sampling
        new_assignment = state.assignment.at[action].set(jnp.logical_or(is_valid, state.assignment[action]))

        # create action mask
        # infeasible_nodes = state.adjacency_matrix[new_assignment].sum(0) >= 1 # not jit compatible
        infeasible_nodes = jnp.where(new_assignment[:, None], state.adjacency_matrix, False).any(0)
        # assert (infeasible_nodes == (state.adjacency_matrix[new_assignment].sum(0) >= 1)).all()
        
        action_mask = jnp.logical_or(infeasible_nodes, new_assignment)

        # update is_done
        num_infeasible_nodes = jax.ops.segment_sum(jnp.asarray(action_mask, dtype=jnp.int32), state.graph_ids, num_segments=state.problem.n_node.shape[0])
        is_done = state.problem.n_node == num_infeasible_nodes 

        # create state and obs
        new_state = MisState(problem=state.problem, assignment=new_assignment, is_done=is_done, adjacency_matrix=state.adjacency_matrix, graph_masks=state.graph_masks, graph_ids=state.graph_ids)

        action_masks = jnp.logical_or(~state.graph_masks, action_mask)
        new_obs = MisObservation(problem=state.problem, assignment=new_assignment, action_masks=action_masks)
        return new_state, new_obs