import os
import argparse
import sys
import random
from functools import partial
import time
import timeit
from datetime import timedelta
# os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1" #hopefully temporary bug with xla/nvidia/cuda or whatever, setting the flag works around that

import jax
import jax.numpy as jnp
from jax import disable_jit
from chex import dataclass, PRNGKey, Array 
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
from torch.utils.data import DataLoader
import orbax.checkpoint as ocp
from moco.mis_task_padded import MisTaskFamily, train_task, MisTaskParams
from moco.lopt import MisOptimizer
from moco.utils import jax_has_gpu
from moco.data_utils import MisCollater, MisDataset
from learned_optimization.optimizers.optax_opts import Adam


if __name__ == '__main__':
    # Argument parsing Arguments: data_path, batch_size, num_steps, learning_rate, seed: optional if not given selects the seed randomly,  verbose: bool to print training progress
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-d", help="path to the data", type=str)
    parser.add_argument("--task_batch_size", "-tb", help="batch size", type=int, default=64)
    parser.add_argument("--num_construction_steps", type=int)
    parser.add_argument("--pad_to_pow2", default=False, action="store_true")
    parser.add_argument("--batch_size_eval", "-be", help="batch size", type=int, default=64)
    parser.add_argument("--num_steps", "-n", help="number of training steps", type=int, default=200)
    parser.add_argument("--seed", "-s", help="seed", type=int, default=random.randrange(sys.maxsize))
    parser.add_argument("--mlflow_uri", help="mlflow uri", type=str, default="logs/mlruns")
    parser.add_argument("--experiment", help="experiment name", type=str, default="mis_eval")
    parser.add_argument("--num_parallel_heatmaps", "-nh", help="number of parallel heatmaps being optimized", type=int, default=1)
    parser.add_argument("--checkpoint_folder", "-cf", help="folder to load checkpoint from", type=str, default=None)
    parser.add_argument("--learning_rate", "-l", help="learning rate for Adam inseat no learned optimizer is used", type=float, default=1e-2)
    parser.add_argument("--chunk_size", help="chunk size for chunking the computation in case of oom", type=int, default=None)
    args = parser.parse_args()
    # pretty print arguments
    print("Arguments:")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    # test gpu
    print("jax has gpu:", jax_has_gpu())

    key = jax.random.PRNGKey(args.seed)

    # load from checkpoint if available
    if args.checkpoint_folder is not None:
        restore_options = ocp.CheckpointManagerOptions(
        best_mode='max',
        best_fn=lambda x: x['val_best_reward'] if 'val_best_reward' in x else x['val_gap_avg'],
    )
        restore_mngr = ocp.CheckpointManager(
            args.checkpoint_folder,
            ocp.PyTreeCheckpointer(),
            options=restore_options)
        
        metadata = restore_mngr.metadata()
        args.top_k = metadata['top_k']

        lopt = MisOptimizer(embedding_size=metadata['embedding_size'], num_layers_init=metadata['num_layers_init'], num_layers_update=metadata['num_layers_update'], aggregation=metadata['aggregation'])
        optimizer_params = restore_mngr.restore(restore_mngr.best_step())
        optimizer = lopt.opt_fn(optimizer_params, is_training=False)
        print(f"Running optimizer from checkpoint {args.checkpoint_folder} step {restore_mngr.best_step()}")
    else:
        args.top_k = 32
        optimizer = Adam(learning_rate=args.learning_rate)
        print(f"Running Adam with lr {args.learning_rate}")
    
    dataset = MisDataset(root=args.data_path)
    collator = MisCollater(pad_to_pow2=args.pad_to_pow2)
    dataset_size = len(dataset)

    task_family = MisTaskFamily(dataset = None, inner_batch_size=args.task_batch_size, unroll_length=args.num_construction_steps, top_k=args.top_k)
    data_loader = DataLoader(dataset, batch_size=args.batch_size_eval, shuffle=True, collate_fn=collator)
    print("Dataset size: ", dataset_size)


    @partial(jax.jit, static_argnums=(2,))
    def parallel_inference(problem, key: PRNGKey, chunk_size=None):
        """Train a task from multiple starting nodes"""
        # create task params
        task_params = MisTaskParams(problem = problem)

        # train task in parallel
        keys = jax.random.split(key, args.num_parallel_heatmaps)

        results = []
        n = args.num_parallel_heatmaps
        if chunk_size is None:
            chunk_size = n

        n_chunks, residual = divmod(n, chunk_size)
        assert residual == 0, "num_parallel_heatmaps must be divisible by chunk_size"
        for i in range(n_chunks):
            chunk_keys = keys[i*chunk_size:(i+1)*chunk_size]
            res = jax.vmap(train_task, in_axes=(None, 0, None, None, None))(task_params, chunk_keys, args.num_steps, optimizer, task_family)
            results.append(res)

        results = jax.tree_map(lambda *x: jnp.concatenate(x), *results)
        return results

    compute_start = time.time()
    
    metrics = []
    for i, batch in tqdm(enumerate(data_loader)):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch.nodes.shape[0])
        res = jax.vmap(parallel_inference, in_axes=(0,0,None))(batch, keys, args.chunk_size)
        metrics.append(res)

    # aggregate batches of results
    # results = {key:jnp.mean(jnp.concatenate([val[key] for val in metrics], axis=0), axis=0) for key in metrics[0].keys()}
    stacked_best_rewards = jnp.concatenate([val['best_reward'] for val in metrics], axis=0)
    best_rewards = stacked_best_rewards.max(axis=1)
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
        mlflow.log_metrics({'max_construction_steps': best_rewards.max().item()})
        mlflow.log_metrics({'total_compute_time': compute_end - compute_start})

    last_best_reward = results['best_reward'][-1]
    print(f"Last best reward: {last_best_reward}")