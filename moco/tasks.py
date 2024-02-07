from typing import Mapping, Tuple
import jax
# from jax import config
# config.update("jax_debug_nans", True)
import jax.numpy as jnp
from jax import disable_jit
import rlax # if importing rlax after learned_optimization, it will throw an error regarding tensorflow probability version?
import jraph
import flax
import numpy as np
import chex
from chex import dataclass, PRNGKey, Array
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union
from functools import partial

from learned_optimization.tasks.base import Batch, ModelState, PRNGKey, Params
from learned_optimization.tasks import base as tasks_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.learned_optimizers import common, base as lopt_base

from moco.environments import CustomTSP, CustomGraphTSP
from jumanji.environments.routing.tsp.generator import Generator, UniformGenerator
from moco.data_utils import sample_tsp
from moco.tsp_actors import nearest_neighbor, HeatmapActor, knn_graph, fully_connected_graph, SparseHeatmapActor
from moco.rl_utils import random_actor, greedy_actor, rollout, random_initial_position, greedy_rollout, pomo_rollout, entmax_actor, entmax_policy_gradient_loss

@dataclass
class TspTaskParams:
    coordinates: Array
    starting_node: int

@dataclass
class ModelState:
    graph: jraph.GraphsTuple
    # best_reward: Array
    top_k_rewards: Array
    top_k_solutions: Array
    step: Array
    # rolling_reward: Array
    # abusing the model state here to pass aux info to the optimizer since the optimizer doesnt get passed the auxs for some reason

@dataclass
class TspMetrics:
    pgl: Array
    mean_reward: Array
    best_reward: Array
    reward_std: Array

class TspTaskFamily(tasks_base.TaskFamily):
    def __init__(self, problem_size: int, batch_size: int, k: int, baseline: Optional[str] = None, causal: bool = False, meta_loss_type: str = 'best', top_k: int = 32, heatmap_init_strategy: str = 'constant', normalize_advantage=False, rollout_actor: str = 'softmax'):
        super().__init__()
        self.datasets = None
        self.problem_size = problem_size
        self.batch_size = batch_size
        self.k = k
        # self.exp_beta = exp_beta
        self.baseline = baseline # options = None (no baseline), 'avg' (average reward), 'best' (best reward)
        self.causal = causal
        self.normalize_advantage=normalize_advantage
        self.meta_loss_type = meta_loss_type # options = 'best', 'difference', TODO: 'log'
        self.heatmap_init_strategy = heatmap_init_strategy # options = 'constant', 'heuristic'
        self.rollout_actor = rollout_actor # options = 'softmax', 'entmax', 'greedy', 'random'
        # with 'best' meta loss at each step is the best so far found solution,
        # with 'difference' meta loss is the difference between the best so far found solution and the best solution in the batch of trajectories
        # clipped at 0, such that the sum of the meta losses over the time steps reflects the best found solution over the optimization steps
        self.top_k = top_k
        assert top_k <= batch_size, "top k must be smaller equal to batch size, otherwise currently the relative gaps include -inf in the first step since not all top k solutions are filled"

    def __repr__(self) -> str:
        return f"TspTaskFamily_n{self.problem_size}_k{self.k}_b{self.batch_size}"

    def sample(self, key: PRNGKey) -> TspTaskParams:
        return TspTaskParams(coordinates=sample_tsp(key, self.problem_size), starting_node=0)

    def task_fn(self, task_params: TspTaskParams) -> tasks_base.Task:
        # TaskParams represent the problem instance (coordinates of the cities in this case)
        problem_size = self.problem_size
        num_edges = self.problem_size * self.k
        k = self.k
        heatmap_model = SparseHeatmapActor(problem_size=problem_size, num_edges=num_edges)
        env = CustomGraphTSP(k, generator=UniformGenerator(problem_size))
        batch_size = self.batch_size
        problem = task_params.coordinates
        starting_node = task_params.starting_node
        # exp_baseline_decay = self.exp_beta
        baseline = self.baseline
        causal = self.causal
        normalize_advantage=self.normalize_advantage
        meta_loss_type = self.meta_loss_type
        top_k = self.top_k
        heatmap_init_strategy = self.heatmap_init_strategy
        rollout_actor = self.rollout_actor
        if rollout_actor == 'softmax':
            actor = random_actor
        elif rollout_actor == 'entmax':
            actor = entmax_actor
        else:
            raise NotImplementedError

        class _Task(tasks_base.Task):

            def _rollout(self, params, rng):
                """Rollout the policy on the problem instance and return the final state, the logits of the policy and intermediate results in order to calculate the pgl loss and other things needed"""
                # init_pos_key, rollout_key = jax.random.split(rng, 2)
                # initial_pos = jax.random.randint(init_pos_key, (batch_size,), 0, problem_size)
                initial_pos = jnp.full((batch_size,), starting_node, dtype=jnp.int32)
                rollout_keys = jax.random.split(rng, batch_size)

                batched_rollout = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None, None, None))
                final_state, (logits, timesteps) = batched_rollout(rollout_keys, problem, initial_pos, params, heatmap_model.apply, actor, env, problem_size-1)
                return final_state, logits, timesteps
            
            def _compute_advantages(self, r, best_reward=None):
                total_reward = r.sum(axis=-1, keepdims=True)
                mean_reward = total_reward.mean()
                cum_reward = r[:,::-1].cumsum(-1)[:,::-1]
                if causal:
                    if baseline == 'avg':
                        adv = cum_reward - cum_reward.mean(axis=0)
                    elif baseline == 'best':
                        raise NotImplementedError
                    elif baseline is None:
                        adv = cum_reward
                    else:
                        raise ValueError(f'baseline {baseline} not implemented')
                    if normalize_advantage:
                        adv = (adv - adv.mean(axis=0)) / (adv.std(axis=0) + 1e-8)
                else:
                    if baseline == 'avg':
                        adv =  total_reward - mean_reward
                    elif baseline == 'best':
                        adv =  total_reward - best_reward
                    elif baseline is None:
                        adv =  total_reward
                    else:
                        raise ValueError(f'baseline {baseline} not implemented')
                    if normalize_advantage:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                return adv 
            
            def _pgl(self, final_state, logits, timesteps , best_reward=None):
                # we need to change in the logits all rows with only -inf entries to 0 to avoid nan loss since distrax.softmax returns inf if the whole row is -inf
                all_inf_rows = jnp.all(logits == -jnp.inf, axis=-1)
                # broadcast row mask and use jnp.where to set all rows with only -inf entries to 0
                logits = jnp.where(jnp.expand_dims(all_inf_rows, 2), 0., logits)
                # logits = logits.at[all_inf_rows].set(jnp.zeros((logits.shape[-1])))
                #  also need to avoid the actions being set to -1 since this is an invalid category in the categorical dist also leading ot -inf log prob
                actions = final_state.trajectory[:,1:]
                actions = jnp.where(actions==-1, 1, actions)
                weights = jnp.roll(timesteps.discount, 1).at[:,0].set(1.)
                weighted_reward = weights * timesteps.reward
                advantages = self._compute_advantages(weighted_reward, best_reward)
                # single_pgl = rlax.policy_gradient_loss(logits[0], actions[0], advantages[0], weights[0])
                if rollout_actor == 'softmax':
                    pgl = jax.vmap(rlax.policy_gradient_loss)(logits, actions, advantages, weights).mean(axis=0)
                elif rollout_actor == 'entmax':
                    pgl = jax.vmap(entmax_policy_gradient_loss)(logits, actions, advantages, weights).mean(axis=0)
                return pgl
            
            def _compute_metrics(self, timesteps, pgl):
                weights = jnp.roll(timesteps.discount, 1).at[:,0].set(1.)
                weighted_reward = (weights * timesteps.reward).sum(-1)
                return TspMetrics(
                        pgl=pgl, 
                        mean_reward=weighted_reward.mean()*-1, 
                        best_reward=weighted_reward.max()*-1, 
                        reward_std= weighted_reward.std())
            
            def init_with_state(self, key: PRNGKey) -> Tuple[Params, ModelState]:
                # initial state is mostly static so use numpy to create since we dont want to trace but evaluate at compile time and then reuse
                graph = knn_graph(problem, k)
                aux_graph = jraph.GraphsTuple(
                    nodes = {'initial_pos': jnp.zeros((problem_size, 1)).at[starting_node].set(1.)}, # binary feature for the starting node
                    edges = {'distances': graph.edges.reshape((num_edges, -1)), 'top_k_sols': np.ones((num_edges, top_k), dtype=np.int32)},
                    globals = {
                        # 'best_cost': np.array([[1.]]), 
                        # 'mean_cost': np.array([[1.]]), 
                        'gaps': np.ones((1, top_k)), 
                        'relative_improvement': np.array([[1.]])
                        },
                    receivers = graph.receivers,
                    senders = graph.senders,
                    n_node = graph.n_node,
                    n_edge = graph.n_edge
                )
                top_k_rewards = np.full((top_k,), -jnp.inf)
                top_k_solutions = np.zeros((top_k, problem_size), dtype=np.int32)
                step = np.array(0)

                initial_state = ModelState(graph=aux_graph, 
                                    # best_reward=jnp.array(-jnp.inf), 
                                    top_k_rewards=top_k_rewards, 
                                    top_k_solutions=top_k_solutions, 
                                    step=step
                                    )
                # initialize heatmap
                init_state, _ = env.reset_from_problem(problem)
                param_dict = heatmap_model.init(key, init_state)
                if heatmap_init_strategy == 'constant':
                    heatmap_params = param_dict

                elif heatmap_init_strategy == 'heuristic':
                    param_dict = flax.core.unfreeze(param_dict)
                    param_dict['params']['heatmap'] = -init_state.graph.edges.reshape(problem_size,k)
                    heatmap_params = flax.core.freeze(param_dict)
                return heatmap_params, initial_state

            def loss_with_state_and_aux(self, params: Params, state: ModelState, key: PRNGKey, _: Any, with_metrics=False) -> Tuple[Array, ModelState, Mapping[str, Array]]:
                # compute the pgl for gradient, meta loss and additional information for the optimizer
                final_state, logits, timesteps = self._rollout(params, key)

                # compute reward
                weights = jnp.roll(timesteps.discount, 1).at[:,0].set(1.)
                weighted_reward = (weights * timesteps.reward).sum(-1)

                # compute top k solutions
                stacked_rewards = jnp.concatenate([state.top_k_rewards, weighted_reward])
                stacked_sols = jnp.concatenate([state.top_k_solutions, final_state.trajectory])
                top_k_inds = jnp.argpartition(stacked_rewards*-1, kth=top_k)[:top_k]
                top_k_rewards = stacked_rewards[top_k_inds]
                top_k_solutions = stacked_sols[top_k_inds]

                # sort top k solutions by reward
                sort_inds = jnp.argsort(top_k_rewards*-1)
                top_k_rewards = top_k_rewards[sort_inds]
                top_k_solutions = top_k_solutions[sort_inds]

                # turn top_k solutions to graph features
                senders = final_state.graph.senders[-1]
                receivers = final_state.graph.receivers[-1]
                edges = final_state.graph.edges[-1]
                # compute best solution as a (sparse) graph
                # idea: broadcast each index of the solution to the senders and receivers and combine them with logical and
                # this obtains a boolean mask over the broadcasted matrix of shape [sol_length, num_edges] 
                # where the row represents the source node and the column the index of the edge list (the real target index thus could be looked up in the receiver list at that column position)
                # summing the mask over the rows thus gives a boolean mask of the edges in the graph, whether they are part of the solution
                convert_sols_fn = jax.vmap(lambda solution: jnp.logical_and(receivers == jnp.roll(solution, -1)[:,None], senders == solution[:,None]).sum(0))
                top_k_sols_as_graph = convert_sols_fn(top_k_solutions)

                # compute pgl with baseline
                global_best_reward = top_k_rewards[0]
                pgl = self._pgl(final_state, logits, timesteps, global_best_reward)

                # compute meta loss and update model_state variable containing additional information for the optimizer and the rolling reward
                previous_best = state.top_k_rewards[0]
                if meta_loss_type == 'best':
                    meta_loss = global_best_reward*-1
                elif meta_loss_type == 'log':
                    meta_loss = jnp.log(global_best_reward*-1)
                elif meta_loss_type == 'difference':
                    meta_loss = jax.lax.cond(
                        previous_best == -jnp.inf,
                        lambda: global_best_reward*-1,
                        lambda: jnp.minimum(0, previous_best - global_best_reward))

                # add aux features and constuct jraph graph
                # relative improvement over last step
                rel_impr = jax.lax.cond(
                    previous_best == -jnp.inf,
                    lambda: jnp.asarray(1.),
                    lambda: (previous_best - global_best_reward)/previous_best)

                # compute relative gaps within top_k solutions
                # TODO: what should happen if the batch size is smaller than top k since initial top k is filled with -inf. currently the relative gaps include -inf in the first step
                gaps = (top_k_rewards - global_best_reward)/global_best_reward

                aux_graph = jraph.GraphsTuple(
                    nodes = {'initial_pos': jnp.zeros((problem_size, 1)).at[starting_node].set(1.)}, 
                    edges = {
                        'distances': edges.reshape((num_edges, -1)), 
                        'top_k_sols': top_k_sols_as_graph.transpose(1,0)
                        },
                    globals = {
                        # 'best_cost': jnp.atleast_2d(global_best_reward*-1), 
                        # 'mean_cost': jnp.atleast_2d(weighted_reward.mean()*-1), 
                        'gaps': jnp.atleast_2d(gaps), 
                        'relative_improvement': jnp.atleast_2d(rel_impr)
                        },
                    receivers = receivers,
                    senders = senders,
                    n_node = final_state.graph.n_node[-1],
                    n_edge = final_state.graph.n_edge[-1]
                )
                # construct aux dict and state
                aux = dict(meta_loss=meta_loss)
                state = ModelState(
                    graph=aux_graph,
                    top_k_rewards=top_k_rewards, 
                    top_k_solutions=top_k_solutions, 
                    step=state.step+1
                    )

                if with_metrics:
                    metrics = self._compute_metrics(timesteps, pgl)
                    return pgl, state, aux, metrics
                else:
                    return pgl, state, aux

            def __repr__(self) -> str:
                return f"TspTask(n={self.num_nodes} k={self.k} b={self.batch_size})"
        return _Task()
    
    def dummy_model_state(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(42), 2)
        cfg = self.sample(key1)
        task = self.task_fn(cfg)
        params, state = task.init_with_state(key2)
        del params
        return state

# @chex.assert_max_traces(n=1)
@partial(jax.jit, static_argnames=['num_steps', 'optimizer', 'task_family'])
def train_task(cfg, key, num_steps, optimizer, task_family):
    """ Train a task for num_steps using the optimizer and return the best reward over the training steps.
    Args:
        optimizer_params: parameters of the optimizer
        cfg: task configuration
        key: random key
        num_steps: number of training steps
        optimizer: optimizer
        task_family: task family
    Returns:
        best_reward: best reward over the training steps
    """

    task = task_family.task_fn(cfg)
    key, params_key, rollout_key = jax.random.split(key, 3)
    params, model_state = task.init_with_state(params_key)

    opt_state = optimizer.init(params, model_state, num_steps=num_steps)

    def loss_fn(param, model_state, key, data):
        """Wrapper around loss_with_state_and_aux to return 2 values."""
        loss, model_state, aux, metrics = task.loss_with_state_and_aux(param, model_state, key, data, with_metrics=True)
        return loss, (model_state, aux, metrics)
        
    def step_fn(state, key):
        opt_state, model_state = state
        params = optimizer.get_params(opt_state)
        grad_fun = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (model_state, aux, metrics)), grad = grad_fun(params, model_state, key, None)
        opt_state = optimizer.update(opt_state, grad, loss, model_state)
        return (opt_state, model_state), metrics

    def run_n_step(opt_state, model_state, key, n):
        random_keys = jax.random.split(key, n)
        (opt_state, model_state), rollout = jax.lax.scan(step_fn, (opt_state, model_state), random_keys)
        return rollout

    results= run_n_step(opt_state, model_state, rollout_key, num_steps)

    best_rewards = jax.lax.associative_scan(jax.numpy.minimum, results['best_reward']) # equivalent to np.minimum.accumulate()which is not yet implemented in jax https://github.com/google/jax/issues/11281
    results.best_reward = best_rewards
    return results