# Moco: A learnable Meta Optimizer for Combinatorial Optimization

## Installing the dependencies
Best from a fresh virtual environment, first install jax for your hardware (Cpu, GPU, TPU). We used version 0.4.23 with python 3.10.11 (3.10.0 does not work). For installation, refer to the documentation, but for example run the following on linux for Nvidia gpus:
```
pip install --upgrade "jax[cuda12]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
```
After installing jax, install the remaining dependencies
```
pip install -r requirements.txt
```
Unfortunately, you have to manually uninstall oryx afterwards:
```
pip uninstall oryx
```
Otherwise the package causes problems, and it is automatically installed alongside another package.

Finally, install our repository as a package locally. Run the following command from this folder
```
pip install -e .
```
## Download the data and checkpoints
Download the trained model checkpoints: [checkpoints](https://drive.google.com/file/d/1G5aN0I5bhDHA5bHmnHBdVJE1rtTM1Ww4/view?usp=sharing)

Download the MIS data: [MIS](https://drive.google.com/file/d/1Ruhx4uXvEY4rZW111fOEgaoywdQeKHEf/view?usp=sharing)

Download the TSP data: [TSP](https://drive.google.com/file/d/1HLDU6aWsdHVfpu55-UJCYdbP6mtQK1nK/view?usp=sharing)

Extract the data into the data folder and the models into the checkpoints folder

For the SATLIB data and used split, refer to [Dimes](https://github.com/DIMESTeam/DIMES)

## Train a TSP model
Train a model for the TSP with b=32 and K=50
```
python experiments/tsp_meta_train.py --problem_size 100 --task_batch_size 32 --max_length 50 --causal --baseline avg --parallel_tasks_train 128 --outer_lr 1e-3 --outer_train_steps 30000 --parallel_tasks_val 128 --val_path your_path/data/tsp/val-100-coords.npy --model_save_path checkpoints
```

## Train a MIS model
Train a model for the MIS with b=32 and K=50
```
python experiments/mis_meta_train.py --train_dataset your_path/data/mis/er_train_700_800_015 --task_batch_size 32 --num_construction_steps 49 --parallel_tasks_train 64 --outer_lr 1e-3 --outer_train_steps 20000 --max_length 50 --parallel_tasks_val 64 --val_dataset your_path/data/mis/er_val_700_800_015 --model_save_path checkpoints
```

## Evaluate a TSP model
Evaluate a TSP model with b=32, K=200 and M=32
```
python experiments/evaluate_tsp.py --data_path your_path/data/tsp/test-100-coords.npy --task_batch_size 32 --batch_size_eval 1 --num_steps 200 --num_starting_nodes 32 --checkpoint_folder your_path/checkpoints/tsp100_200_32
```
with 2-opt:
```
python experiments/evaluate_tsp.py --data_path your_path/data/tsp/test-100-coords.npy --task_batch_size 32 --batch_size_eval 32 --num_steps 200 --num_starting_nodes 1 --checkpoint_folder your_path/checkpoints/tsp100_200_32_ls --two_opt_t_max 10000
```
## Evaluate a MIS model
Evaluate a MIS model with b=32, K=200 and M=32
```
python experiments/evaluate_mis.py --data_path your_path/data/mis/er_test_700_800_015 --task_batch_size 32 --num_construction_steps 50 --batch_size_eval 1 --num_steps 200 --num_parallel_heatmaps 32 --checkpoint_folder your_path/checkpoints/mis_er_200_32
```

## Displaying results
We use mlflow for tracking. You can look at the logs by
```
cd logs
mlflow ui
```