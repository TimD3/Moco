from typing import Optional, Sequence, Tuple, Union, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
import chex
from chex import PRNGKey, Array, dataclass # standard dataclass doesnt work with jax since its not a valid jax type, they seem to use it in jumanjki, dont know how it works, maybe it got explicitly registered but dont know where they do that
import jraph
from jraph import GraphsTuple

@dataclass
class MisState:
    problem: jraph.GraphsTuple
    assignment: Array
    is_done: bool
    # edge_padding_mask: Array
    node_padding_mask: Array
    adjacency_matrix: Array

@dataclass
class MisObservation:
    problem: jraph.GraphsTuple
    assignment: Array
    action_mask: Array # 1 where action is invalid

class MaxIndependentSet:
    def reset_from_problem(self, problem):
        """reset the environment to a new problem, the problem is a jraph.GraphsTuple that must be padded with at least one additional node by jraph.pad_with_graphs, since this is required for jraphs padding utilities to work"""
        num_nodes = problem.nodes.shape[0]
        node_padding_mask = jraph.get_node_padding_mask(problem)
        # edge_padding_mask = jraph.get_edge_padding_mask(padded_graph)
        assignment = jnp.zeros(num_nodes, dtype=jnp.bool_)
        is_done = jnp.zeros((), dtype=jnp.bool_)
        adjacency = jnp.zeros((num_nodes,num_nodes), dtype=jnp.bool_)
        adjacency = adjacency.at[problem.senders, problem.receivers].set(1)
        state = MisState(problem=problem, assignment=assignment, is_done=is_done, node_padding_mask=node_padding_mask, adjacency_matrix=adjacency)
        action_mask = ~node_padding_mask
        obs = MisObservation(problem=problem, assignment=assignment, action_mask=action_mask)
        return state, obs

    def step(self, state, action):
        """step the environment with the given action, action is the index of the node to add to the independent set"""
        # is_infeasible = action_infeasibility(state, action)
        
        new_assignment = jax.lax.cond(
            state.is_done, 
            lambda x,a: x, 
            lambda x,a: x.at[a].set(True),
            state.assignment, action)
        
        infeasible_nodes = jnp.where(new_assignment[:, None], state.adjacency_matrix, False).any(0)
        action_mask = infeasible_nodes | ~state.node_padding_mask | new_assignment
        is_done = action_mask.all()

        new_state = MisState(problem=state.problem, assignment=new_assignment, is_done=is_done, node_padding_mask=state.node_padding_mask, adjacency_matrix=state.adjacency_matrix)

        new_obs = MisObservation(problem=state.problem, assignment=new_assignment, action_mask=action_mask)
        return new_state, new_obs