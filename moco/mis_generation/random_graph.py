from moco.mis_generation.generator import DataGenerator
import networkx as nx
import random
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import os
import functools
# from logzero import logger
# from utils import run_command_with_live_output

class GraphSampler(ABC):
    @abstractmethod
    def generate_graph(self):
        pass

class ErdosRenyi(GraphSampler):
    def __init__(self, min_n, max_n, p):
        self.min_n = min_n
        self.max_n = max_n
        self.p = p

    def __str__(self):
        return f"ER_{self.min_n}_{self.max_n}_{self.p}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.erdos_renyi_graph(n, self.p)

class BarabasiAlbert(GraphSampler):
    def __init__(self, min_n, max_n, m):
        self.min_n = min_n
        self.max_n = max_n
        self.m = m

    def __str__(self):
        return f"BA_{self.min_n}_{self.max_n}_{self.m}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.barabasi_albert_graph(n, min(self.m, n))


class HolmeKim(GraphSampler):
    def __init__(self, min_n, max_n, m, p):
        self.min_n = min_n
        self.max_n = max_n
        self.m = m
        self.p = p

    def __str__(self):
        return f"HK_{self.min_n}_{self.max_n}_{self.m}_{self.p}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.powerlaw_cluster_graph(n, min(self.m, n), self.p)


class WattsStrogatz(GraphSampler):
    def __init__(self, min_n, max_n, k, p):
        self.min_n = min_n
        self.max_n = max_n
        self.k = k
        self.p = p

    def __str__(self):
        return f"WS_{self.min_n}_{self.max_n}_{self.k}_{self.p}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.watts_strogatz_graph(n, self.k, self.p)

class RandomGraphGenerator(DataGenerator):
    def __init__(self, output_path, graph_sampler: GraphSampler, num_graphs = 1):
        self.num_graphs = num_graphs
        self.output_path = output_path
        self.graph_sampler = graph_sampler

    def generate(self, gen_labels = False, weighted = False):
        stubs = []

        for i in range(self.num_graphs):
            stub = f"{self.graph_sampler}_{i}"
            stubs.append(stub)

        imap_unordered_bar(functools.partial(self.func, gen_labels=gen_labels, weighted=weighted),
                           stubs, n_processes=32)

    def func(self, stub, gen_labels, weighted):
        G = self.graph_sampler.generate_graph()
        if weighted:
            weight_mapping = {vertex: int(weight) for vertex, weight in
                              zip(G.nodes, self.random_weight(G.number_of_nodes(), sigma=30, mu=100))}
            nx.set_node_attributes(G, values=weight_mapping, name='weight')

        status = ""
        output_file = self.output_path / f"{stub}{'.non-optimal' if gen_labels and status != 'Optimal' else ''}.gpickle"
        with open(output_file, "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)


from multiprocessing import Pool
from tqdm import *


def imap_unordered_bar(func, args, n_processes=2):
    p = Pool(n_processes)
    args = list(args)
    print(args)
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
    pbar.close()
    p.close()
    p.join()
