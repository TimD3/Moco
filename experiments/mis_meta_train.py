import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import disable_jit
from functools import partial
import rlax
from tqdm import tqdm
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
from torch.utils.data import DataLoader
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

from moco.data_utils import *
from moco.utils import *
from moco.mis_task_padded import MisTaskFamily, train_task, MisTaskParams
from moco.lopt import MisOptimizer
import mlflow
    

if __name__ == "__main__":

    ################## config ##################
    parser = argparse.ArgumentParser()
    # TaskFamily and problem setting
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--task_batch_size", type=int)
    parser.add_argument("--num_construction_steps", type=int) # how many sequential construction steps a single rollout does on MIS. Ideally it should stop when all environments in the batch is done but because of jax.grad needs static boundaries and thus is incompatible with jax lax while we need to give a fixed predertimend num steps t run the the heatmap over the environment
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--pad_to_pow2", default=False, action="store_true")

    # meta loss
    parser.add_argument("--meta_loss_type", type=str, choices=["best", "difference", "log"], default="best")

    # metaTraining
    parser.add_argument("--parallel_tasks_train", type=int)
    parser.add_argument("--outer_lr", type=float)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--outer_train_steps", type=int, default=10000)
    parser.add_argument("--gradient_estimator", type=str, choices=["full_es", "pes"], default="full_es")
    parser.add_argument("--trunc_length", type=int, default=10) # truncation length for pes training
    parser.add_argument("--trunc_schedule", type=str, choices=["constant", "loguniform", "piecewise_linear"], default="constant") # for full es and pes training
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--max_length", type=int) # number of steps to unroll the optimizer on the inner task
    parser.add_argument("--piecewise_linear_fraction", type=float, default=0.2, help="fraction of outer_train_steps after which the truncation length is max_length")
    parser.add_argument("--patience", type=int, default=20) # early stopping
    parser.add_argument("--dont_stack_antithetic", action="store_true") # whether to stack antithetic samples for gradient estimation
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--lr_schedule", type=str, choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--num_devices", type=int, default=None)
    parser.add_argument("--clip_loss_diff", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=0.01)
    # parser.add_argument("--mp_policy", type=str, choices=["params=float32,compute=float16,output=float32", "params=float32,compute=bfloat16,output=float32", "params=float32,compute=float32,output=float32"], default="params=float32,compute=float32,output=float32")

    # meta optimizer
    parser.add_argument("--aggregation", type=str, choices=["sum", "max"], default="max")
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--num_layers_init", type=int, default=3)
    parser.add_argument("--num_layers_update", type=int, default=3)
    parser.add_argument("--checkpoint_folder", "-cf", help="folder to load checkpoint from", type=str, default=None)
    # parser.add_argument("--ignore_feature_names", nargs="+", default=['best_cost', 'mean_cost'])

    # validation and logging
    parser.add_argument("--parallel_tasks_val", type=int)
    # parser.add_argument("--artifact_path", type=str, default='/home/tim/Documents/research/artifacts')
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--model_save_path", type=str, default="checkpoints")
    parser.add_argument("--val_steps", type=int, default=200)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--mlflow_uri", type=str, default="logs/mlruns")
    parser.add_argument("--experiment_name", type=str, default="meta_mis")
    parser.add_argument("--disable_tqdm", default=False, action="store_true")
    parser.add_argument("--ood_dataset", default=None, type=str)
    parser.add_argument("--ood_num_construction_steps", type=int, default=None)

    # debug
    parser.add_argument("--disable_jit", default=False, action="store_true")
    args = parser.parse_args()

    # current time as string for saving in human readable format
    # start_time = datetime.now().strftime("%m%d%Y-%H%M%S")
    if args.model_save_path is not None:
        unique_filename = str(uuid.uuid4())
        args.model_save_path = os.path.join(args.model_save_path, unique_filename)

    if args.num_devices is None:
        args.num_devices = len(jax.devices())

    # assert args.trunc_length <= args.max_length, "trunc_length must be smaller than max_length"
    assert args.min_length <= args.max_length, "loguniform_trunc_min must be smaller equal than max_length"
    assert args.parallel_tasks_train % args.num_devices == 0, f"parallel_tasks_train must be divisible by num_devices {args.num_devices}, jax_devices: {jax.devices()}"
    assert args.parallel_tasks_val % args.num_devices == 0, f"parallel_tasks_val must be divisible by num_devices {args.num_devices}, jax_devices: {jax.devices()}"
    
    # test gpu # TODO: switch to chex 
    print("jax has gpu:", jax_has_gpu())
    ################## config ##################
    print("Loading dataset...")

    dataset = MisDataset(root=args.train_dataset)
    collator = MisCollater(pad_to_pow2=args.pad_to_pow2)
    dataset_size = len(dataset)
    loader = DataLoader(dataset, batch_size=dataset_size, shuffle=False, collate_fn=collator, drop_last=False, num_workers=0)
    data = next(iter(loader))
    # data = jax.tree_util.tree_map(lambda x: np.array(x), data)
    print("Done loading dataset...")

    task_family = MisTaskFamily(dataset = data, inner_batch_size=args.task_batch_size, unroll_length=args.num_construction_steps, top_k=args.top_k)
    val_loader = DataLoader(MisDataset(root=args.val_dataset), batch_size=args.parallel_tasks_val, shuffle=True, collate_fn=MisCollater(pad_to_pow2=args.pad_to_pow2))
    # learnable  optimizers
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
        # overwrite args with metadata from checkpoint that affects the optimizer
        args.embedding_size = metadata['embedding_size']
        args.num_layers_init = metadata['num_layers_init']
        args.num_layers_update = metadata['num_layers_update']
        args.aggregation = metadata['aggregation']
        print("Warning: Overwriting args with metadata from checkpoint that affects the optimizer")

        lopt = MisOptimizer(embedding_size=args.embedding_size, num_layers_init=args.num_layers_init, num_layers_update=args.num_layers_update, aggregation=args.aggregation)
        optimizer_params = restore_mngr.restore(restore_mngr.best_step())
        class ParamInitializer(MetaInitializer):
            def __init__(self, params):
                self.params = params
            def init(self, key: chex.PRNGKey):
                return self.params
        theta_init = ParamInitializer(optimizer_params)

        print(f"Running Mis optimizer from checkpoint {args.checkpoint_folder} step {restore_mngr.best_step()}")
    
    # otherwise initialize from scratch
    else:
        dummy_model_state = task_family.dummy_model_state()
        lopt = MisOptimizer(embedding_size=args.embedding_size, num_layers_init=args.num_layers_init, num_layers_update=args.num_layers_update, aggregation=args.aggregation, dummy_observation=dummy_model_state)
        theta_init = lopt
        print(f"Running optimizer from scratch")

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
    
    def truncated_es_estimator(task_family):
        truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
            task_family,
            lopt,
            truncation_schedule.NeverEndingTruncationSchedule(),
            num_tasks=args.parallel_tasks_train,
            meta_loss_with_aux_key="meta_loss",
            task_name=str(task_family)
        )
        return truncated_es.TruncatedES(
            truncated_step = truncated_step,
            unroll_length=args.trunc_length,
            stack_antithetic_samples=not args.dont_stack_antithetic,
            steps_per_jit=10,
            std=args.sigma
        )
                 
    def truncated_pes_estimator(task_family):
        truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
            task_family,
            lopt,
            trunc_sched,
            num_tasks=args.parallel_tasks_train,
            random_initial_iteration_offset=args.max_length,
            meta_loss_with_aux_key="meta_loss",
            task_name=str(task_family)
            )
        return truncated_pes.TruncatedPES(
            truncated_step=truncated_step, trunc_length=args.trunc_length, steps_per_jit=10, stack_antithetic_samples=not args.dont_stack_antithetic, std=args.sigma)
    
    estimators = {
        "full_es": full_es_estimator,
        "pes": truncated_pes_estimator
    }
    
    gradient_estimators = [
        estimators[args.gradient_estimator](task_family),
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
            decay_steps=args.outer_train_steps - args.warmup_steps,
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
        best_mode='max',
        best_fn=lambda x: x['val_best_reward'],
    )
    metadata = dict(vars(args))
    # metadata['subset'] = str(metadata['subset']) # to avoid json serialization error
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
        for i, batch in enumerate(dataset):
            # print(jax.tree_map(lambda x: x.shape, batch))
            batch_size = batch.nodes.shape[0]
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, batch_size)
            
            batched_task_params = MisTaskParams(problem=batch)
            if args.num_devices > 1 and batch_size % args.num_devices == 0: # only use pmap if batch size is divisible by num_devices, usually the last batch is not and then we use jit on single device
                res = unsplit(train_task_pmap(split(batched_task_params), split(keys), args.max_length, opt, task_family))
                print(f"pmap with {args.num_devices} devices and batch size {batch_size}")
            else:
                res = train_task_jit(batched_task_params, keys, args.max_length, opt, task_family)
                print("jit with 1 device and batch size", batch_size)
            metrics.append(res)
        # aggregate batches of results
        results = {key:jnp.mean(jnp.concatenate([val[key] for val in metrics], axis=0), axis=0) for key in metrics[0].keys() if key != 'max_steps'}
        results['max_steps'] = jnp.max(jnp.concatenate([val['max_steps'] for val in metrics], axis=0), axis=0)

        if aggregate:
            results = {
                'val_mean_mean_reward' : jnp.mean(results['mean_reward']).item(),
                'val_mean_last_reward': results['mean_reward'][-1].item(),
                'val_best_reward':results['best_reward'][-1].item(),
                'val_mean_reward_std':jnp.mean(results['reward_std']).item(),
                'val_max_steps': results['max_steps'].max().item(),
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

        for i in tqdm(range(args.outer_train_steps), disable=args.disable_tqdm):
            # print(f"start outer step {i}", flush=True)
            # validation
            if i % args.val_steps == 0:
                key, subkey = jax.random.split(key)
                theta = outer_trainer.get_meta_params(outer_trainer_state)
                results = validate(val_loader, task_family, theta, subkey)
                mlflow.log_metrics(results, step=i)

                # checkpointing
                mngr.save(i, hk.data_structures.to_mutable_dict(theta), metrics=results)

                # early stopping
                early_stop_score = -results['val_best_reward']
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
            # print(f"stop outer step {i}", flush=True)

        # training done, log final validation metrics
        best_val_parameters = hk.data_structures.to_haiku_dict(mngr.restore(mngr.best_step()))
        key, subkey = jax.random.split(key)
        
        results = validate(val_loader, task_family, best_val_parameters, subkey, aggregate=True)
        mlflow.log_metrics(results, step=args.outer_train_steps)

        # log ood
        if args.ood_dataset is not None:
            ood_loader = DataLoader(MisDataset(root=args.ood_dataset), batch_size=args.parallel_tasks_val, shuffle=True, collate_fn=MisCollater(pad_to_pow2=args.pad_to_pow2))
            ood_family = MisTaskFamily(dataset = None, inner_batch_size=args.task_batch_size, unroll_length=args.ood_num_construction_steps, top_k=args.top_k)
            key, subkey = jax.random.split(key)
            ood_results = validate(ood_loader, ood_family, best_val_parameters, subkey, aggregate=True)
            mlflow.log_metrics({'ood_score': ood_results['val_best_reward']}, step=0)
        print("Finished training", flush=True)