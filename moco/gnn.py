# GNN in Jax
import jax
import jax.numpy as jnp

import chex
from chex import Array
import jraph
from jraph import GraphsTuple, GraphNetwork
from jraph._src.utils import segment_sum, segment_max
import haiku as hk
# import jmp

class GNN(hk.Module):

  def __init__(self, 
    num_layers: int = 5,
    embedding_size: int = 64,
    aggregation = 'max',
    embed_globals = True,
    update_globals = True,
    decode_globals = False,
    decode_edges = True,
    decode_edge_dimension = 1,
    decode_global_dimension: int = 1,
    normalization = 'none', # pre, post, none
    name="GNN"):

    super().__init__(name=name)
    self.num_layers = num_layers
    self.embedding_size = embedding_size
    self.aggregation = aggregation
    self.embed_globals = embed_globals
    self.update_globals = update_globals
    self.decode_globals = decode_globals
    self.decode_edges = decode_edges
    self.decode_edge_dimension = decode_edge_dimension
    self.decode_global_dimension = decode_global_dimension
    self.normalization = normalization

    if self.aggregation == 'max':
       self.aggregate_fn = segment_max
    elif self.aggregation == 'sum':
       self.aggregate_fn = segment_sum
    

  def __call__(self, graph: jraph.GraphsTuple) -> jraph.ArrayTree:

    def update_global_fn(
        aggregated_nodes,
        aggregated_edges,
        globals_
    ):
        concatenated_features = jnp.concatenate([aggregated_nodes, aggregated_edges, globals_], axis=-1)
        transformed_global = hk.Linear(output_size=self.embedding_size, name='global_fn_linear')(concatenated_features)
        transformed_global = jax.nn.relu(transformed_global)
        return globals_ + transformed_global
    
    def update_edge_fn(
        edge_features,
        sender_features,
        receiver_features,
        globals_
    ) -> Array:
        """Edge update function for the GNN."""
        if self.embed_globals:
            concatenated_features = jnp.concatenate([edge_features, sender_features, receiver_features, globals_], axis=-1)
        else:
            concatenated_features = jnp.concatenate([edge_features, sender_features, receiver_features], axis=-1)
        transformed_edge = hk.Linear(output_size=self.embedding_size, name='edge_fn_linear')(concatenated_features)
        ln = hk.LayerNorm(axis=0, create_scale=True, create_offset=True, name='edge_fn_ln')

        if self.normalization == 'pre':
            transformed_edge = ln(transformed_edge)
            transformed_edge = jax.nn.relu(transformed_edge)
        elif self.normalization == 'post':
            transformed_edge = jax.nn.relu(transformed_edge)
            transformed_edge = ln(transformed_edge)
        elif self.normalization == 'none':
            transformed_edge = jax.nn.relu(transformed_edge)
        else:
            raise ValueError(f"Unknown normalization {self.normalization}")
        return edge_features + transformed_edge

    def update_node_fn(
        node_features, 
        sender_features, 
        receiver_features, 
        globals_
    ) -> Array:
        """Node update function for the GNN."""
        if self.embed_globals:
            concatenated_features = jnp.concatenate([node_features, sender_features, receiver_features, globals_], axis=-1)
        else: 
            concatenated_features = jnp.concatenate([node_features, sender_features, receiver_features], axis=-1)    
        transformed_node = hk.Linear(output_size=self.embedding_size, name='node_fn_linear')(concatenated_features)
        ln = hk.LayerNorm(axis=0, create_scale=True, create_offset=True, name='node_fn_ln')

        if self.normalization == 'pre':
            transformed_node = ln(transformed_node)
            transformed_node = jax.nn.relu(transformed_node)
        elif self.normalization == 'post':
            transformed_node = jax.nn.relu(transformed_node)
            transformed_node = ln(transformed_node)
        elif self.normalization == 'none':
            transformed_node = jax.nn.relu(transformed_node)
        else:
            raise ValueError(f"Unknown normalization {self.normalization}")
        return node_features + transformed_node

    embedding = jraph.GraphMapFeatures(
        embed_edge_fn=hk.Linear(output_size=self.embedding_size, name='edge_embedding'),
        embed_node_fn=hk.Linear(output_size=self.embedding_size, name='node_embedding'),
        embed_global_fn=hk.Linear(output_size=self.embedding_size, name='global_embedding') if self.embed_globals else None,
        )
    
    graph = embedding(graph)

    graph = hk.Sequential([
        GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn if self.update_globals and self.embed_globals else None,
            aggregate_edges_for_globals_fn=self.aggregate_fn,
            aggregate_nodes_for_globals_fn=self.aggregate_fn,
            aggregate_edges_for_nodes_fn=self.aggregate_fn,
        )
        for _ in range(self.num_layers)
    ], name='GNN_Blocks')(graph)

    if self.decode_globals:
        decoded_globals = hk.Linear(output_size = self.decode_global_dimension, name="global_decoder")(graph.globals)
        graph = graph._replace(globals = decoded_globals)
    
    if self.decode_edges:
        decoded_egdes = hk.Linear(output_size = self.decode_edge_dimension, name="edge_decoder")(graph.edges)
        graph = graph._replace(edges = decoded_egdes)

    return graph