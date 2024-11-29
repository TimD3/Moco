# GNN in Jax
from typing import Any, Callable, Optional, Tuple, Union
import jax
import jax.numpy as jnp

import chex
from chex import Array
import jraph
from jraph import GraphsTuple, GraphNetwork, GraphConvolution
from jraph._src.utils import segment_sum, segment_max
import haiku as hk
# import jmp

class GCNLayer(hk.Module):
    def __init__(self, embedding_size, aggregation, symmetric_normalization, update_globals, name: str | None = None):
        super().__init__(name)

        self.embedding_size = embedding_size
        self.aggregation = aggregation
        # self.normalization = normalization
        self.symmetric_normalization = symmetric_normalization
        self.update_globals = update_globals
        if self.aggregation == 'max':
            self.aggregate_fn = segment_max
        elif self.aggregation == 'sum':
            self.aggregate_fn = segment_sum

        if self.update_globals:
            self.global_fn = GraphNetwork(
                update_edge_fn=None,
                update_node_fn=None,
                update_global_fn=lambda an, ae, g: g + jax.nn.relu(hk.Linear(output_size=self.embedding_size, name='global_fn_linear')(jnp.concatenate([an, g], axis=-1))),
                # aggregate_edges_for_globals_fn=None,
                aggregate_nodes_for_globals_fn=self.aggregate_fn,
                aggregate_edges_for_nodes_fn=None,
            )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.ArrayTree:
        W1 = hk.Linear(output_size = self.embedding_size, name="W1")
        nodes = W1(graph.nodes)
        gcn = GraphConvolution(
            update_node_fn = lambda nf: hk.Linear(output_size = self.embedding_size, name="W2")(nf),
            aggregate_nodes_fn=self.aggregate_fn,
            add_self_edges=False,
            symmetric_normalization=self.symmetric_normalization)
        updated_graph = gcn(graph)
        nodes = nodes + updated_graph.nodes

        # include globals
        if updated_graph.globals is not None:
            W3 = hk.Linear(output_size = self.embedding_size, name="W3")
            transformed_global = W3(updated_graph.globals[0:1])
            nodes = nodes + transformed_global

        nodes = jax.nn.relu(nodes)
        nodes = nodes + graph.nodes # res connection
        updated_graph = updated_graph._replace(nodes = nodes)

        # update globals
        if self.update_globals:
            updated_graph = self.global_fn(updated_graph)
        return updated_graph

class GCN(hk.Module):

  def __init__(self, 
    num_layers: int = 3,
    embedding_size: int = 64,
    aggregation = 'max',
    embed_globals = True,
    update_globals = True,
    decode_globals = False,
    decode_node_dimension = 1,
    decode_global_dimension: int = 1,
    normalization = 'post', # pre, post, none
    symmetric_normalization = True,
    name="GNN"):

    super().__init__(name=name)
    self.num_layers = num_layers
    self.embedding_size = embedding_size
    self.aggregation = aggregation
    self.embed_globals = embed_globals
    self.update_globals = update_globals
    self.decode_globals = decode_globals
    self.decode_node_dimension = decode_node_dimension
    self.decode_global_dimension = decode_global_dimension
    self.normalization = normalization
    self.symmetric_normalization = symmetric_normalization
    
  def __call__(self, graph: jraph.GraphsTuple) -> jraph.ArrayTree:

    embedding = jraph.GraphMapFeatures(
        embed_edge_fn=None,
        embed_node_fn=hk.Linear(output_size=self.embedding_size, name='node_embedding'),
        embed_global_fn=hk.Linear(output_size=self.embedding_size, name='global_embedding') if self.embed_globals else None,
        )
    graph = embedding(graph)

    graph = hk.Sequential([
        GCNLayer(embedding_size=self.embedding_size, aggregation=self.aggregation, symmetric_normalization=self.symmetric_normalization, update_globals=self.update_globals, name = "GCN_Layer")
        for _ in range(self.num_layers)
    ], name='GCN_Blocks')(graph)

    if self.decode_globals:
        decoded_globals = hk.Linear(output_size = self.decode_global_dimension, name="global_decoder")(graph.globals)
        graph = graph._replace(globals = decoded_globals)
    else:
        graph = graph._replace(globals = None)
    decoded_nodes = hk.Linear(output_size = self.decode_node_dimension, name="node_decoder")(graph.nodes)
    graph = graph._replace(nodes = decoded_nodes)
    return graph