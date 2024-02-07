import jax
import jax.numpy as jnp
import numpy as np
import argparse
import tensorflow as tf
import os
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
import jraph
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union, List
from moco.mis_generation.random_graph import imap_unordered_bar
import time
from multiprocessing import Pool

def load_and_process(filename):
    print(f"Processing {filename}")
    data = nx.read_gpickle(filename)
    print(f"Converting to Pytorch Geometric")
    data = from_networkx(data)
    if data.num_node_features == 0:
        data.x = torch.ones(data.num_nodes, 1)
    return data

class MisDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, num_processes=1):
        self.num_processes = num_processes
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        instances = os.listdir(self.root)
        instances = [instance for instance in instances if instance.endswith(".gpickle")]
        return instances

    @property
    def processed_file_names(self):
        return ['mis.pt']

    def process(self):
        # Read data into huge `Data` list.
        start = time.time()
        data_list = []
        files = [os.path.join(self.root, filename) for filename in self.raw_file_names]
        # use multiprocessing to speed up processing
        with Pool(self.num_processes) as p:
            data_list = p.map(load_and_process, files)
        
        # for filename in self.raw_file_names:
        #     print(f"Processing {filename}")
        #     data = nx.read_gpickle(os.path.join(self.root, filename))
        #     print(f"Converting to Pytorch Geometric")
        #     data = from_networkx(data)
        #     if data.num_node_features == 0:
        #         data.x = torch.ones(data.num_nodes, 1)
        #     data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        print(f"Saving {len(data_list)} graphs to {self.processed_paths[0]}")
        self.save(data_list, self.processed_paths[0])
        end = time.time()
        print(f"Processing took {end - start} seconds")

def pyg_to_jraph(graph):
    """convert a pytorch geometric graph to a jraph graph"""
    jraph_graph = jraph.GraphsTuple(
        n_node=jnp.array([graph.num_nodes]),
        n_edge=jnp.array([graph.num_edges]),
        nodes=jnp.array(graph.x),
        edges=jnp.array(graph.edge_index),
        senders=jnp.array(graph.edge_index[0]),
        receivers=jnp.array(graph.edge_index[1]),
    )

    return jraph_graph

def sample_tsp(key, problem_size):
    """Sample a random TSP problem.
    Args:
    key: a JAX PRNGKey.
    problem_size: the number of cities in the TSP problem.
    Returns:
    coordinates: an array of shape [problem_size, 2] containing the coordinates of the cities.
    """
    coordinates = jax.random.uniform(key, shape=(problem_size, 2))
    return coordinates

def load_data(path, batch_size, subset=None):
    """Load a dataset from a .npy file.
    Args:
    path: path to the .npy file.
    batch_size: batch size for the tf.data.Dataset.
    subset: slice object. optional subset of the dataset to load.
    Returns:
    dataset: a tf.data.Dataset containing the data.
    """
    data = np.load(path)
    if subset is not None:
        data = data[subset]
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
    return dataset

# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y

class MisCollater:
    def __init__(
        self, pad_to_pow2: bool = False
    ):
        self.pad_to_pow2 = pad_to_pow2
        

    def __call__(self, batch: List[Any]) -> Any:
        max_num_nodes, max_num_edges = 0, 0
        for i in range(len(batch)):
            max_num_nodes = max(max_num_nodes, batch[i].num_nodes)
            max_num_edges = max(max_num_edges, batch[i].num_edges)
        
        pad_nodes_to = _nearest_bigger_power_of_two(max_num_nodes + 1) if self.pad_to_pow2 else max_num_nodes + 1 # pad at least one node
        pad_edges_to = _nearest_bigger_power_of_two(max_num_edges) if self.pad_to_pow2 else max_num_edges
        for i in range(len(batch)):
            jraph_tuple = pyg_to_jraph(batch[i])
            padded_graph = jraph.pad_with_graphs(jraph_tuple, pad_nodes_to, pad_edges_to, 2)
            batch[i] = padded_graph

        # concatenate graphs
        batch = stack_jraph_graphs(batch)
        return batch

def stack_jraph_graphs(graphs: Sequence[jraph.GraphsTuple]) -> jraph.GraphsTuple:
    """stack a sequence of graphs into a single graph with batched attributes. All graphs need to be padded to the same number of nodes and edges.
    Args:
    graphs: a sequence of jraph.GraphsTuple.
    Returns:
    graph: a jraph.GraphsTuple containing the stacked graphs.
    """
    nodes = jnp.stack([graph.nodes for graph in graphs], axis=0) if graphs[0].nodes is not None else None
    edges = jnp.stack([graph.edges for graph in graphs], axis=0) if graphs[0].edges is not None else None
    globals_ = jnp.stack([graph.globals for graph in graphs], axis=0) if graphs[0].globals is not None else None
    senders = jnp.stack([graph.senders for graph in graphs], axis=0) if graphs[0].senders is not None else None
    receivers = jnp.stack([graph.receivers for graph in graphs], axis=0) if graphs[0].receivers is not None else None
    n_node = jnp.stack([graph.n_node for graph in graphs], axis=0)
    n_edge = jnp.stack([graph.n_edge for graph in graphs], axis=0)
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=globals_,
    )

def pyg_to_jraph(graph):
    if graph.batch is not None:
        graphs = jraph.GraphsTuple(
            nodes=np.array(graph.x),
            edges=None, # this particular data doesn't have edge features, hence we set to None
            globals=None,
            n_node=np.array(graph.ptr[1:] - graph.ptr[0:-1]),
            n_edge=np.array(torch.logical_and((graph.edge_index[0][:,None] < graph.ptr[1:]),(graph.edge_index[0][:,None] >= graph.ptr[:-1])).sum(0)),
            senders=np.array(graph.edge_index[0]),
            receivers=np.array(graph.edge_index[1])
        )
    else:
        graphs = jraph.GraphsTuple(
            nodes=np.array(graph.x),
            edges=None, # this particular data doesn't have edge features, hence we set to None
            globals=None,
            n_node=np.expand_dims(np.asarray(graph.num_nodes), 0),
            n_edge=np.expand_dims(np.asarray(graph.num_edges), 0),
            senders=np.array(graph.edge_index[0]),
            receivers=np.array(graph.edge_index[1])
        )
    return graphs

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", type=str, help="type of problem to generate data for", choices=["tsp", "mis"])
    # for TSP
    parser.add_argument("--problem_size", type=int, help="size of the problem (number of cities in TSP)")
    parser.add_argument("--num_samples", type=int, help="number of samples to generate")
    parser.add_argument("--path", help="path to the output file")
    parser.add_argument("--seed", type=int, help="seed for the random number generator")
    # for MIS
    parser.add_argument("--data_path", type=str, help="path to the output file")
    parser.add_argument("--min_n", type=int, help="min number of nodes")
    parser.add_argument("--max_n", type=int, help="max number of nodes")
    parser.add_argument("--edge_probability", type=float, help="edge probability")
    parser.add_argument("--num_samples", type=int, help="number of samples to generate")

    args = parser.parse_args()
    
    if args.problem_type == "tsp":
        # create and save data
        key = jax.random.PRNGKey(args.seed)
        data = jax.vmap(sample_tsp, in_axes=(0, None))(jax.random.split(key, args.num_samples), args.problem_size)
        data = np.array(data)
        print(f'Saving data of shape {data.shape} to {args.path}')
        np.save(args.path, data)

    elif args.problem_type == "mis":

        from moco.mis_generation.random_graph import RandomGraphGenerator
        from moco.mis_generation.random_graph import ErdosRenyi
        from moco.data_utils import MisDataset
        from pathlib import Path
        data_path = Path(args.data_path) #"/home/tim/Documents/research/data/mis/er_train_tiny"
        sampler = ErdosRenyi(args.min_n, args.max_n, args.edge_probability)
        generator = RandomGraphGenerator(data_path, graph_sampler = sampler, num_graphs = args.num_samples)
        generator.generate()
        # get number threads
        num_processes = os.cpu_count()
        dataset = MisDataset(data_path, num_processes=num_processes)
    
