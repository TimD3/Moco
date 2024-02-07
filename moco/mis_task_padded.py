import distrax
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

from moco.rl_utils import random_actor, greedy_actor, rollout, random_initial_position, greedy_rollout, pomo_rollout, entmax_actor, entmax_policy_gradient_loss

from torch.utils.data import DataLoader
from moco.tsp_actors import MisActor
from moco.data_utils import MisDataset, pyg_to_jraph, MisCollater
from moco.mis_env import MaxIndependentSet
from moco.rl_utils import mis_rollout

@dataclass
class MisTaskParams:
    problem: jraph.GraphsTuple

@dataclass
class ModelState:
    graph: jraph.GraphsTuple
    # best_reward: Array
    top_k_rewards: Array
    top_k_solutions: Array
    step: Array
    # abusing the model state here to pass aux info to the optimizer since the optimizer doesnt get passed the auxs for some reason

@dataclass
class MisMetrics:
    pgl: Array
    mean_reward: Array
    best_reward: Array
    reward_std: Array
    max_steps: Array

class MisTaskFamily(tasks_base.TaskFamily):
    def __init__(self, dataset: jraph.GraphsTuple, inner_batch_size: int, unroll_length: int, meta_loss_type: str = 'best', top_k: int = 32):
        super().__init__()
        self.datasets = None
        self.unroll_length = unroll_length # number of steps to unroll the policy needs to be static because of jax lax.while cant be used with grad
        self.inner_batch_size = inner_batch_size # number of rollouts per problem
        # self.outer_batch_size = outer_batch_size # number of problems per batch
        self.meta_loss_type = meta_loss_type # options = 'best', 'log'
        self.top_k = top_k
        if dataset is not None:
            self.dataset_size = dataset.nodes.shape[0]
            self.callback = lambda idx: jax.tree_map(lambda x: jnp.array(x[idx]), dataset)
            self.result_shape = jax.tree_map(lambda x: jax.ShapeDtypeStruct(x[0].shape, x.dtype), dataset)

        assert top_k <= inner_batch_size, "top k must be smaller equal to batch size, otherwise currently the relative gaps include -inf in the first step since not all top k solutions are filled"

    def __repr__(self) -> str:
        return f"MisTaskFamily"

    def sample(self, key: PRNGKey) -> MisTaskParams:
        # random_index = jax.random.randint(key, shape=(), minval=0, maxval=self.dataset_size-1)
        # np_random_index = np.random.randint(0, high=, size=None, dtype=int)
        # problem = jax.tree_map(lambda x: x[], self.data)
        # problem = jax.tree_map(lambda x: x[random_index], self.data)
        idx = jax.random.randint(key, (), 0, self.dataset_size)
        problem = jax.pure_callback(self.callback, self.result_shape, idx, vectorized=True)
        return MisTaskParams(problem=problem)

    def task_fn(self, task_params: MisTaskParams) -> tasks_base.Task:
        num_nodes = task_params.problem.nodes.shape[0]
        env = MaxIndependentSet()
        # state, obs = env.reset_from_problem(jraph_batch)
        heatmap_model = MisActor(problem_size=num_nodes)
        # heatmap_params = heatmap.init(key, obs) 
        inner_batch_size = self.inner_batch_size
        # outer_batch_size = self.outer_batch_size
        problem = task_params.problem
        meta_loss_type = self.meta_loss_type
        top_k = self.top_k
        actor = random_actor
        unroll_length = self.unroll_length

        class _Task(tasks_base.Task):

            def _rollout(self, params, rng):
                """Rollout the policy on the problem instance and return the final state, the logits of the policy and intermediate results in order to calculate the pgl loss and other things needed"""
                rollout_keys = jax.random.split(rng, inner_batch_size)
                batched_rollout = jax.vmap(mis_rollout, in_axes=(0, None, None, None, None, None, None))
                final_state, info = batched_rollout(rollout_keys, problem, params, heatmap_model.apply, actor, env, unroll_length)
                return final_state, info
                # return mis_rollout(rng, problem, params, heatmap_model.apply, actor, env, unroll_length)
            
            def _compute_advantages(self, final_state):
                """Compute the advantages for the policy gradient loss"""
                rewards = final_state.assignment.sum(1)
                advantages = (rewards - rewards.mean(0))
                return rewards, advantages
            
            @staticmethod
            @partial(jax.vmap, in_axes=(0, 0, 0, 0))
            def _pgl(a_t, logits_t, adv, w_t):
                log_pi_a_t = distrax.Softmax(logits_t, 1.).log_prob(a_t)
                adv = jax.lax.stop_gradient(adv)
                loss_per_timestep = -log_pi_a_t * adv * w_t
                return loss_per_timestep.sum() / w_t.sum()
            
            def init_with_state(self, key: PRNGKey) -> Tuple[Params, ModelState]:
                problem_size = problem.nodes.shape[0]
                aux_graph = jraph.GraphsTuple(
                    nodes = {'dummy': problem.nodes , 'top_k_sols': np.zeros((problem_size, top_k), dtype=np.float32)}, # binary feature for the starting node
                    edges = problem.edges,
                    globals = {
                        'gaps': np.ones((2, top_k)), # 2 because graph is padded with one extra graph
                        'relative_improvement': np.ones((2, 1)) # 2 because graph is padded with one extra graph
                        },
                    receivers = problem.receivers,
                    senders = problem.senders,
                    n_node = problem.n_node,
                    n_edge = problem.n_edge
                )
                top_k_rewards = np.full((top_k,), -jnp.inf)
                top_k_solutions = np.zeros((top_k, problem_size), dtype=np.float32)
                step = np.array(0)

                initial_state = ModelState(
                    graph=aux_graph, 
                    # best_reward=jnp.array(-jnp.inf), 
                    top_k_rewards=top_k_rewards, 
                    top_k_solutions=top_k_solutions, 
                    step=step
                    )
                # initialize heatmap
                init_state, init_obs = env.reset_from_problem(problem)
                param_dict = heatmap_model.init(key, init_obs)
                return param_dict, initial_state

            def loss_with_state_and_aux(self, params: Params, state: ModelState, key: PRNGKey, _: Any, with_metrics=False) -> Tuple[Array, ModelState, Mapping[str, Array]]:
                # compute the pgl for gradient, meta loss and additional information for the optimizer
                final_state, info = self._rollout(params, key)
                logits, is_done, actions = info
                rewards, advantages = self._compute_advantages(final_state)
                weights = jnp.asarray(~is_done, dtype=jnp.float32)
                pgl = self._pgl(actions, logits, advantages, weights).mean()

                # # compute top k solutions
                stacked_rewards = jnp.concatenate([state.top_k_rewards, rewards])
                stacked_sols = jnp.concatenate([state.top_k_solutions, final_state.assignment.astype(jnp.float32)])
                top_k_inds = jax.lax.top_k(stacked_rewards, top_k)[1]
                top_k_rewards = stacked_rewards[top_k_inds]
                top_k_solutions = stacked_sols[top_k_inds]

                # sort top k solutions by reward
                sort_inds = jnp.argsort(-top_k_rewards)
                top_k_rewards = top_k_rewards[sort_inds]
                top_k_solutions = top_k_solutions[sort_inds]

                # compute meta loss
                global_best_reward = top_k_rewards[0]
                if meta_loss_type == 'best':
                    meta_loss = -global_best_reward
                elif meta_loss_type == 'log':
                    meta_loss = -jnp.log(global_best_reward)
                aux = dict(meta_loss=meta_loss)

                # add aux features and constuct jraph graph
                # relative improvement over last step
                previous_best = state.top_k_rewards[0]
                rel_impr = jax.lax.cond(
                    previous_best == -jnp.inf,
                    lambda: jnp.asarray(1.),
                    lambda: (previous_best - rewards.max())/previous_best)

                # # compute relative gaps within top_k solutions
                # # TODO: what should happen if the batch size is smaller than top k since initial top k is filled with -inf. currently the relative gaps include -inf in the first step
                gaps = (top_k_rewards - global_best_reward)/global_best_reward

                aux_graph = jraph.GraphsTuple(
                    nodes = {'dummy': state.graph.nodes['dummy'] , 'top_k_sols': top_k_solutions.transpose((1,0))}, # binary feature for the starting node
                    edges = state.graph.edges,
                    globals = {
                        'gaps': jnp.repeat(jnp.atleast_2d(gaps), 2, axis=0, total_repeat_length=2), # need to repeat globals for the padded graph
                        'relative_improvement': jnp.repeat(jnp.atleast_2d(rel_impr), 2, axis=0, total_repeat_length=2)
                        },
                    receivers = state.graph.receivers,
                    senders = state.graph.senders,
                    n_node = state.graph.n_node,
                    n_edge = state.graph.n_edge
                )
                # construct state
                aux = dict(meta_loss=meta_loss)
                state = ModelState(
                    graph=aux_graph,
                    top_k_rewards=top_k_rewards, 
                    top_k_solutions=top_k_solutions, 
                    step=state.step+1
                    )
                if with_metrics:
                    # compute metrics
                    metrics = MisMetrics(
                        pgl = pgl,
                        mean_reward = rewards.mean(),
                        best_reward = rewards.max(),
                        reward_std = rewards.std(),
                        max_steps = rewards.max()
                        )
                    return pgl, state, aux, metrics
                else:
                    return pgl, state, aux
            def __repr__(self) -> str:
                return f"TspTask(n={self.num_nodes} k={self.k} b={self.batch_size})"
        return _Task()
    
    def dummy_model_state(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(42), 2)
        cfg = self.sample(key1)
        # cfg = jax.tree_map(lambda x:x[0], cfg) # remove batch dimension
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

    results = run_n_step(opt_state, model_state, rollout_key, num_steps)

    best_rewards = jax.lax.associative_scan(jax.numpy.maximum, results['best_reward']) # equivalent to np.maximum.accumulate()which is not yet implemented in jax https://github.com/google/jax/issues/11281
    results.best_reward = best_rewards
    return results