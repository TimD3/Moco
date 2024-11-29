import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import disable_jit
from functools import partial
import rlax
import tqdm
import matplotlib.pyplot as plt
from functools import partial
import argparse
import chex
from chex import dataclass, PRNGKey, Array
from flax.training.early_stopping import EarlyStopping
import orbax.checkpoint as ocp
import haiku as hk
import optax
import uuid
# import jmp

from datetime import datetime

from learned_optimization.outer_trainers import full_es, truncated_pes, truncated_es, gradient_learner, truncation_schedule, lopt_truncated_step

from learned_optimization.tasks import quadratics, base as tasks_base
from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks.datasets import base as datasets_base

from learned_optimization.learned_optimizers import base as lopt_base, mlp_lopt
from learned_optimization.optimizers import learning_rate_schedules, base as opt_base
from learned_optimization.optimizers.optax_opts import Adam, SGD, SGDM, RMSProp, AdamW, OptaxOptimizer
from learned_optimization.outer_trainers.gradient_learner import MetaInitializer

from learned_optimization import optimizers, eval_training
# import oryx
# from learned_optimization import summary
# assert summary.ORYX_LOGGING, "Oryx logging not working"

from moco.environments import CustomTSP as TSP
from moco.data_utils import *
from moco.plot_utils import plot_tsp_grid
from moco.tsp_actors import nearest_neighbor, HeatmapActor
from moco.rl_utils import random_actor, greedy_actor, rollout, random_initial_position, greedy_rollout, pomo_rollout
from moco.utils import *
from moco.tasks import TspTaskFamily, train_task, TspTaskParams
from moco.lopt import HeatmapOptimizer
import mlflow
    

if __name__ == "__main__":

    ################## config ##################
    parser = argparse.ArgumentParser()
    # TaskFamily and problem setting
    parser.add_argument("--problem_size", type=int)
    parser.add_argument("--task_batch_size", type=int, help="b in the paper")
    parser.add_argument("--max_length", type=int, default=50, help="Budget K in the paper") # number of steps to unroll the optimizer on the inner task
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--heatmap_init_strategy", type=str, choices=["heuristic", "constant"], default="heuristic") # doesnt get used, the initialization is overwritten in the learned optimizer
    parser.add_argument("--rollout_actor", type=str, choices=["softmax", "entmax"], default="softmax")
    parser.add_argument("--two_opt_t_max", type=int, default=None)
    parser.add_argument("--first_accept", action="store_true")

    # pgl calculation
    parser.add_argument("--causal", "-c", help="use causal accumulation of rewards for policy gradient calc", action="store_true")
    parser.add_argument("--baseline", "-b", help="specify baseline for policy gradient calc", type=str, default="avg", choices=[None, "avg"])

    # meta loss
    parser.add_argument("--meta_loss_type", type=str, choices=["best", "log"], default="best")

    # metaTraining
    parser.add_argument("--parallel_tasks_train", type=int)
    parser.add_argument("--outer_lr", type=float)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--outer_train_steps", type=int, default=10000)
    parser.add_argument("--trunc_schedule", type=str, choices=["constant", "loguniform", "piecewise_linear"], default="constant") # for full es and pes training
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--piecewise_linear_fraction", type=float, default=0.2, help="fraction of outer_train_steps after which the truncation length is max_length")
    parser.add_argument("--patience", type=int, default=20) # early stopping
    parser.add_argument("--dont_stack_antithetic", action="store_true") # whether to stack antithetic samples for gradient estimation
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--lr_schedule", type=str, choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--num_devices", type=int, default=None)
    parser.add_argument("--clip_loss_diff", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=0.01)

    # meta optimizer
    parser.add_argument("--update_strategy", type=str, choices=["direct", "temperature"], default="temperature")
    parser.add_argument("--aggregation", type=str, choices=["sum", "max"], default="sum")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--num_layers_init", type=int, default=3)
    parser.add_argument("--num_layers_update", type=int, default=3)
    parser.add_argument("--normalization", type=str, choices=["pre", "post", "none"], default="post")
    parser.add_argument("--checkpoint_folder", "-cf", help="folder to load checkpoint from", type=str, default=None)

    # validation and logging
    parser.add_argument("--parallel_tasks_val", type=int)
    parser.add_argument("--val_path", type=str)
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--val_steps", type=int, default=200)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--mlflow_uri", type=str, default="logs")
    parser.add_argument("--experiment_name", type=str, default="meta_tsp")
    parser.add_argument("--disable_tqdm", default=False, action="store_true")
    parser.add_argument("--ood_path", default=None, type=str)

    # debug
    parser.add_argument("--disable_jit", default=False, action="store_true")
    parser.add_argument("--subset", type=parse_slice, default=None, help="slice of the val set to use for validation (e.g. 0:16) only for debugging")

    args = parser.parse_args()

    # current time as string for saving in human readable format
    # start_time = datetime.now().strftime("%m%d%Y-%H%M%S")
    if args.model_save_path is not None:
        unique_filename = str(uuid.uuid4())
        args.model_save_path = os.path.join(args.model_save_path, unique_filename)

    if args.num_devices is None:
        args.num_devices = len(jax.devices())

    assert args.min_length <= args.max_length, "loguniform_trunc_min must be smaller equal than max_length"
    assert args.parallel_tasks_train % args.num_devices == 0, f"parallel_tasks_train must be divisible by num_devices {args.num_devices}, jax_devices: {jax.devices()}"
    assert args.parallel_tasks_val % args.num_devices == 0, f"parallel_tasks_val must be divisible by num_devices {args.num_devices}, jax_devices: {jax.devices()}"
    
    # test gpu # TODO: switch to chex 
    print("jax has gpu:", jax_has_gpu())
    ################## config ##################

    val_dataset = load_data(args.val_path, batch_size=args.parallel_tasks_val, subset=args.subset)
    task_family = TspTaskFamily(args.problem_size, args.task_batch_size, args.k, baseline = args.baseline, causal = args.causal, meta_loss_type = args.meta_loss_type, top_k=args.top_k, heatmap_init_strategy=args.heatmap_init_strategy, rollout_actor=args.rollout_actor, two_opt_t_max=args.two_opt_t_max, first_accept=args.first_accept)

    # load from checkpoint if available
    if args.checkpoint_folder is not None:
        restore_options = ocp.CheckpointManagerOptions(
        best_mode='min',
        best_fn=lambda x: x['val_last_best_reward'] if 'val_last_best_reward' in x else x['val_gap_avg'],
    )
        restore_mngr = ocp.CheckpointManager(
            args.checkpoint_folder,
            ocp.PyTreeCheckpointer(),
            options=restore_options)
        
        metadata = restore_mngr.metadata()
        # overwrite args with metadata from checkpoint that affects the optimizer
        args.embedding_size = metadata['embedding_size']
        args.num_layers_init = metadata['num_layers_init']
        args.num_layers_update = metadata['num_layers_update']
        args.normalize_inputs = metadata['normalize_inputs']
        args.lopt = metadata['lopt']
        args.aggregation = metadata['aggregation']
        args.update_strategy = metadata['update_strategy']
        args.normalization = metadata['normalization']
        print("Warning: Overwriting args with metadata from checkpoint that affects the optimizer")
        
        lopt = HeatmapOptimizer(embedding_size=args.embedding_size, num_layers_init=args.num_layers_init, num_layers_update=args.num_layers_update, aggregation=args.aggregation, update_strategy=args.update_strategy, normalization=args.normalization)
        optimizer_params = restore_mngr.restore(restore_mngr.best_step())
        class ParamInitializer(MetaInitializer):
            def __init__(self, params):
                self.params = params
            def init(self, key: chex.PRNGKey):
                return self.params
        theta_init = ParamInitializer(optimizer_params)

        print(f"Running {metadata['lopt']} optimizer from checkpoint {args.checkpoint_folder} step {restore_mngr.best_step()}")
    
    # otherwise initialize from scratch
    else:
        dummy_model_state = task_family.dummy_model_state()
        lopt = HeatmapOptimizer(embedding_size=args.embedding_size, num_layers_init=args.num_layers_init, num_layers_update=args.num_layers_update, aggregation=args.aggregation, update_strategy=args.update_strategy, dummy_observation=dummy_model_state, normalization=args.normalization)
        theta_init = lopt
        print(f"Running GNN optimizer from scratch")

    if args.trunc_schedule == "constant":
        trunc_sched = truncation_schedule.ConstantTruncationSchedule(args.max_length)
    elif args.trunc_schedule == "loguniform":
        trunc_sched = truncation_schedule.LogUniformLengthSchedule(args.min_length, args.max_length)
    elif args.trunc_schedule == "piecewise_linear":
        sched = learning_rate_schedules.PiecewiseLinear([0, int(args.outer_train_steps * args.piecewise_linear_fraction)], [args.min_length, args.max_length])
        trunc_sched = truncation_schedule.ScheduledTruncationSchedule(sched)
    
    def split(tree):
        """Splits the first axis of `arr` evenly across the number of devices."""
        return jax.tree_map(lambda arr: arr.reshape(args.num_devices, arr.shape[0] // args.num_devices, *arr.shape[1:]), tree)

    def unsplit(tree):
        """Concatenates the first axis of `arr` across all devices."""
        return jax.tree_map(lambda arr: arr.reshape(-1, *arr.shape[2:]), tree)

    def full_es_estimator(task_family):
        truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
            task_family,
            lopt,
            truncation_schedule.NeverEndingTruncationSchedule(),
            num_tasks=args.parallel_tasks_train // args.num_devices,
            meta_loss_with_aux_key="meta_loss",
            task_name=str(task_family)
        )
        if args.num_devices > 1:
            print(f"full es with pmap with {args.num_devices} devices and {args.parallel_tasks_train // args.num_devices} tasks per device")
            return full_es.PMAPFullES(
                truncated_step = truncated_step,
                truncation_schedule = trunc_sched,
                loss_type = "min",
                stack_antithetic_samples=not args.dont_stack_antithetic,
                steps_per_jit=10,
                num_devices=args.num_devices,
                replicate_data_across_devices=False,
                clip_loss_diff=args.clip_loss_diff,
                std=args.sigma
            )
            
        else:
            return full_es.FullES(
                truncated_step = truncated_step,
                truncation_schedule = trunc_sched,
                loss_type = "min",
                stack_antithetic_samples=not args.dont_stack_antithetic,
                steps_per_jit=10,
                clip_loss_diff=args.clip_loss_diff,
                std=args.sigma
            )

    gradient_estimators = [
        full_es_estimator(task_family),
    ]
    print("gradient_estimators:", gradient_estimators)

    # optimizer
    if args.lr_schedule == "constant":
        lr_schedule = optax.constant_schedule(args.outer_lr)
    elif args.lr_schedule == "cosine":
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.outer_lr,
            warmup_steps=args.warmup_steps,
            decay_steps=args.outer_train_steps,
            end_value=0.0,
            )

    theta_opt = optax.chain(
        optax.clip(args.grad_clip) if args.grad_clip is not None else optax.identity(),
        optax.adamw(learning_rate=lr_schedule),
        )
    theta_opt = OptaxOptimizer(theta_opt)
    print('outer opt:', theta_opt)

    # Keeps a maximum of 3 checkpoints and keeps the best one
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        best_mode='min',
        best_fn=lambda x: x['val_last_best_reward'],
    )
    metadata = dict(vars(args))
    metadata['subset'] = str(metadata['subset']) # to avoid json serialization error
    mngr = ocp.CheckpointManager(
        args.model_save_path,
        ocp.PyTreeCheckpointer(),
        options=options,
        metadata=metadata)

    early_stop = EarlyStopping(min_delta=1e-3, patience=args.patience)
    print("early stop:", early_stop)
    print("mngr:", mngr)
    
    outer_trainer = gradient_learner.SingleMachineGradientLearner(
        theta_init, gradient_estimators, theta_opt)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    outer_trainer_state = outer_trainer.init(subkey)

    # debug train_task
    # problem = next(val_dataset.as_numpy_iterator())[0]
    # res = train_task(theta, jnp.array(problem), key)
    # print(res)
    train_task_batched = jax.vmap(train_task, in_axes=(0, 0, None, None, None))
    train_task_jit = jax.jit(train_task_batched, static_argnames=['num_steps', 'optimizer', 'task_family'])
    if args.num_devices > 1:
        train_task_pmap = jax.pmap(train_task_batched, axis_name='data', in_axes=(0, 0, None, None, None), static_broadcasted_argnums=(2,3,4))
    
    def validate(dataset, task_family, optimizer_params, key, aggregate=True):
        """Validate the outer trainer state on a batch of problems from the validation set."""
        opt = lopt.opt_fn(optimizer_params)
        
        metrics = []
        for i, batch in enumerate(dataset.as_numpy_iterator()):
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, batch.shape[0])
            
            # print(jax.tree_map(lambda x: x.shape, (optimizer_params, batch, keys)))
            task_p_fn = jax.vmap(lambda c,s: TspTaskParams(coordinates=c, starting_node=s), in_axes=(0,0))
            batched_task_params = task_p_fn(jnp.array(batch, dtype=jnp.float32), jnp.ones((batch.shape[0],), dtype=jnp.int32))

            if args.num_devices > 1 and batch.shape[0] % args.num_devices == 0: # only use pmap if batch size is divisible by num_devices, usually the last batch is not and then we use jit on single device
                res = unsplit(train_task_pmap(split(batched_task_params), split(keys), args.max_length, opt, task_family))
                print(f"pmap with {args.num_devices} devices and batch size {batch.shape[0]}")
            else:
                res = train_task_jit(batched_task_params, keys, args.max_length, opt, task_family)
                print("jit with 1 device and batch size", batch.shape[0])
            metrics.append(res)
        # aggregate batches of results
        results = {key:jnp.mean(jnp.concatenate([val[key] for val in metrics], axis=0), axis=0) for key in metrics[0].keys()}
        if aggregate:
            results = {
                **{f'val_mean_{key}':jnp.mean(results[key]).item() for key in results}, 
                **{f'val_last_{key}':results[key][-1].item() for key in results}
            }
        return results
    
    # print args
    print("args:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print("Starting training...", flush=True)
    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    experiment = mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as mlflow_cm, disable_jit(args.disable_jit) as jit_cm:

        # log args
        mlflow.log_params(vars(args))
        slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
        if slurm_job_id is not None:
            mlflow.log_param('slurm_job_id', slurm_job_id)
            print(f'Logged slurm_job_id: {slurm_job_id}', flush=True)

        for i in tqdm.tqdm(range(args.outer_train_steps), disable=args.disable_tqdm):
            # validation
            if i % args.val_steps == 0:
                key, subkey = jax.random.split(key)
                theta = outer_trainer.get_meta_params(outer_trainer_state)
                results = validate(val_dataset, task_family, theta, subkey)
                mlflow.log_metrics(results, step=i)

                # checkpointing
                mngr.save(i, hk.data_structures.to_mutable_dict(theta), metrics=results)

                # early stopping
                early_stop_score = results['val_last_best_reward']
                _, early_stop = early_stop.update(early_stop_score)
                if early_stop.should_stop:
                    print('Met early stopping criteria, breaking...')
                    break

            # meta training
            lr = lr_schedule(outer_trainer_state.gradient_learner_state.theta_opt_state.optax_opt_state[1][2].count)
            key, subkey = jax.random.split(key)
            # with jax.profiler.trace("tmp/jax-trace", create_perfetto_link=True):
            outer_trainer_state, loss, metrics = outer_trainer.update(
                outer_trainer_state, subkey, with_metrics=True
                )
            
            if i % args.log_steps == 0:
                # replace '||' with '_' to make it a valid mlflow metric name
                metrics = {k.replace("||", "__"): float(v) for k, v in metrics.items() if not 'collect' in k}
                metrics['lr'] = float(lr)
                mlflow.log_metrics(metrics, step=i)

        # training done, log final validation metrics
        best_val_parameters = hk.data_structures.to_haiku_dict(mngr.restore(mngr.best_step()))
        key, subkey = jax.random.split(key)
        
        results = validate(val_dataset, task_family, best_val_parameters, subkey, aggregate=False)
        aggregates = {
                **{f'val_mean_{key}':jnp.mean(results[key]).item() for key in results}, 
                **{f'val_last_{key}':results[key][-1].item() for key in results}
            }
        mlflow.log_metrics(aggregates, step=args.outer_train_steps)

        # log ood
        if args.ood_path is not None:
            ood_dataset = load_data(args.ood_path, batch_size=args.parallel_tasks_val, subset=args.subset)
            _, ood_size, _ = ood_dataset.element_spec.shape
            ood_family = TspTaskFamily(ood_size, args.task_batch_size, args.k, baseline = args.baseline, causal = args.causal, meta_loss_type = args.meta_loss_type, top_k=args.top_k, two_opt_t_max=args.two_opt_t_max, first_accept=args.first_accept)
            key, subkey = jax.random.split(key)
            ood_results = validate(ood_dataset, ood_family, best_val_parameters, subkey, aggregate=True)
            mlflow.log_metrics({'ood_score': ood_results['val_last_best_reward']}, step=0)