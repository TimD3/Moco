import numpy as np
import mlflow
import jax
import jax.numpy as jnp

from jax._src.lib import pytree
import pickle
from pathlib import Path
from typing import Union

def save_pytree(data: pytree, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_pytree(path: Union[str, Path]) -> pytree:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_jnpz(path, **kwargs):
    """Saves a dict of jnp arrays to a npz file"""
    with open(path, 'wb') as f:
        jnp.savez(f, **kwargs)

def load_jnpz(path):
    """Loads a dict of jnp arrays from a npz file"""
    with open(path, 'rb') as f:
        arrays = dict(jnp.load(f))
    return arrays

def parse_slice(s):
    a = [int(e) if e.strip() else None for e in s.split(":")]
    return slice(*a)

def average_dict_of_time_series(series):
    """"
    series: list of dicts of lists with same keys
    returns: dict of lists with same keys, where each list is the average of the corresponding lists in series
    """
    avg_series = {}
    for key in series[0].keys():
        avg_series[key] = np.mean([s[key] for s in series], axis=0)
    return avg_series

def mlflow_log_dict_of_lists(dict_of_lists):
    """Logs a dict of lists to mlflow"""
    for metric_name, metric_history in dict_of_lists.items():
        for i, val in enumerate(metric_history):
            # print(metric_name, i+1, val)
            mlflow.log_metric(metric_name, val, step=i+1)

def jax_has_gpu():
    """Returns True if jax can find a gpu, False otherwise"""
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        return False

def pytree_repr(pytree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x).shape, pytree)


def dataclass_to_dict_of_lists(dataclass_list, stack=False):
    """given a list of dataclasses, return a dict of lists where each key is a field of the dataclass and each value is a list of the values of that field in the dataclasses"""
    dict_of_lists = {k: [getattr(dc, k) for dc in dataclass_list] for k in dataclass_list[0].__dataclass_fields__.keys()}
    if stack:
        dict_of_lists = {k: jnp.stack(v) for k, v in dict_of_lists.items()}
    return dict_of_lists
