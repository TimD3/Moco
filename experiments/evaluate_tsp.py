import os
import argparse
import sys
import random
from functools import partial
import time
import timeit
from datetime import timedelta

import jax
import jax.numpy as jnp
from jax import disable_jit
from chex import dataclass, PRNGKey, Array 
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow

from moco.tasks import TspTaskFamily, train_task, TspTaskParams
from moco.lopt import HeatmapOptimizer
from moco.utils import jax_has_gpu
from moco.data_utils import load_data
from learned_optimization.optimizers.optax_opts import Adam
from learned_optimization.learned_optimizers import base as lopt_base, mlp_lopt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-d", help="path to the data", type=str)
    parser.add_argument("--task_batch_size", "-tb", help="batch size", type=int, default=64)
    parser.add_argument("--batch_size_eval", "-be", help="batch size", type=int, default=64)
    parser.add_argument("--num_steps", "-n", help="number of training steps", type=int, default=200)
    parser.add_argument("--learning_rate", "-l", help="learning rate", type=float, default=1e-2)
    parser.add_argument("--seed", "-s", help="seed", type=int, default=random.randrange(sys.maxsize))
    parser.add_argument("--verbose", "-v", help="print training progress", action="store_true")
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--heatmap_init_strategy", type=str, choices=["heuristic", "constant"], default="heuristic") # gets overwritten by the model
    parser.add_argument("--rollout_actor", type=str, choices=["softmax", "entmax"], default="softmax")
    parser.add_argument("--k", "-k", help="number of nearest neighbors", type=int, default=None)
    parser.add_argument("--causal", "-c", help="use causal accumulation of rewards for policy gradient calc", action="store_true")
    parser.add_argument("--baseline", "-b", help="specify baseline for policy gradient calc", type=str, default=None, choices=[None, "avg"])
    parser.add_argument("--mlflow_uri", help="mlflow uri", type=str, default="logs")
    parser.add_argument("--experiment", help="experiment name", type=str, default="tsp")
    parser.add_argument("--num_starting_nodes", "-ns", help="number of starting nodes", type=int, default=1)
    parser.add_argument("--checkpoint_folder", "-cf", help="folder to load checkpoint from", type=str, default=None)
    parser.add_argument("--two_opt_t_max", type=int, default=None)
    parser.add_argument("--first_accept", action="store_true")
    args = parser.parse_args()
    # pretty print arguments
    print("Arguments:")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    # test gpu
    print("jax has gpu:", jax_has_gpu())

    key = jax.random.PRNGKey(args.seed)

    dataset = load_data(args.data_path, batch_size=args.batch_size_eval)
    _, problem_size, _ = dataset.element_spec.shape
    dataset_size = sum([i.shape[0] for i in dataset.as_numpy_iterator()])
    print("Dataset size: ", dataset_size, "Problem size: ", problem_size)

    # load optimizer
    if args.checkpoint_folder is not None:
        import orbax.checkpoint as ocp
        restore_options = ocp.CheckpointManagerOptions(
        best_mode='min',
        best_fn=lambda x: x['val_last_best_reward'],
    )
        restore_mngr = ocp.CheckpointManager(
            args.checkpoint_folder,
            ocp.PyTreeCheckpointer(),
            options=restore_options)
        
        metadata = restore_mngr.metadata()
        # overwrite command line arguments with checkpoint metadata
        args.top_k = metadata['top_k']
        args.heatmap_init_strategy = metadata['heatmap_init_strategy']
        args.rollout_actor = metadata['rollout_actor']
        args.k = metadata['k']
        args.causal = metadata['causal']
        args.baseline = metadata['baseline']

        lopts = {
            "adam": lopt_base.LearnableAdam(),
            "gnn": HeatmapOptimizer(embedding_size=metadata["embedding_size"], num_layers_init=metadata["num_layers_init"], num_layers_update=metadata["num_layers_update"], aggregation=metadata["aggregation"], update_strategy=metadata["update_strategy"], normalization=metadata["normalization"])
            }
        l_optimizer = lopts[metadata['lopt']]
        optimizer_params = restore_mngr.restore(restore_mngr.best_step())
        optimizer = l_optimizer.opt_fn(optimizer_params, is_training=False)
        print(f"Running {metadata['lopt']} optimizer from checkpoint {args.checkpoint_folder} step {restore_mngr.best_step()}")

    else:
        optimizer = Adam(learning_rate=args.learning_rate)
        print(f"Running Adam with lr {args.learning_rate}")

    task_family = TspTaskFamily(problem_size, args.task_batch_size, k=args.k, baseline=args.baseline, causal=args.causal, top_k=args.top_k, heatmap_init_strategy=args.heatmap_init_strategy, rollout_actor=args.rollout_actor, two_opt_t_max=args.two_opt_t_max, first_accept=args.first_accept)

    print("Task family: ", task_family)

    @jax.jit
    def train_task_from_multiple_starts(coordinates, key: PRNGKey):
        """Train a task from multiple starting nodes"""
        # create task params
        key, subkey = jax.random.split(key)
        starting_nodes = jax.random.choice(subkey, problem_size, shape=(args.num_starting_nodes,), replace=False) # sample n non repeating starting nodes
        coordinates = jnp.tile(coordinates[None,:,:], (args.num_starting_nodes,1,1)) # repeat coordinates n times
        task_params = TspTaskParams(coordinates=coordinates, starting_node=starting_nodes) # batched task params

        # train task in parallel
        keys = jax.random.split(key, args.num_starting_nodes)
        results = jax.vmap(train_task, in_axes=(0, 0, None, None, None))(task_params, keys, args.num_steps, optimizer, task_family)
        return results
    
    # log compile and single execution time
    single_problem = next(iter(dataset.as_numpy_iterator()))[0]
    key, subkey = jax.random.split(key)
    # measure compile time
    compile_time = timeit.timeit(partial(train_task_from_multiple_starts.lower, single_problem, key)().compile, number=10)/10
    print(f"Compile time: {timedelta(seconds=compile_time)}")
    # measure single instance execution time
    compiled_task = train_task_from_multiple_starts.lower(single_problem, key).compile()
    single_execution_time = timeit.timeit(partial(compiled_task, single_problem, key), number=1)
    print(f"Single execution time: {timedelta(seconds=single_execution_time)}")

    compute_start = time.time()
    
    metrics = []
    for i, batch in tqdm(enumerate(dataset.as_numpy_iterator())):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch.shape[0])
        res = jax.vmap(train_task_from_multiple_starts)(jnp.array(batch, dtype=jnp.float32), keys)
        metrics.append(res)

    # aggregate batches of results
    # results = {key:jnp.mean(jnp.concatenate([val[key] for val in metrics], axis=0), axis=0) for key in metrics[0].keys()}
    stacked_best_rewards = jnp.concatenate([val['best_reward'] for val in metrics], axis=0)
    best_rewards = stacked_best_rewards.min(axis=1)
    results = {'best_reward': jnp.mean(best_rewards, axis=0), 'best_reward_std': jnp.std(best_rewards, axis=0)}

    compute_end = time.time()
    print(f"Compute time: {timedelta(seconds=compute_end - compute_start)}")

    # track runs with mlflow
    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    experiment = mlflow.set_experiment(args.experiment)
    with mlflow.start_run(experiment_id=experiment.experiment_id):

        mlflow.log_params(vars(args))
        for i in range(args.num_steps):
            mlflow.log_metrics({key:val[i].item() for key, val in results.items()}, step=i)
        mlflow.log_metrics({'compile_time': compile_time, 'single_instance_execution_time': single_execution_time, 'total_compute_time': compute_end - compute_start})

    last_best_reward = results['best_reward'][-1]
    print(f"Last best reward: {last_best_reward}")