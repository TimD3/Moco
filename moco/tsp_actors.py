import jax
import jax.numpy as jnp
from jax import lax, random
import jraph
from flax import linen as nn

from jumanji.environments.routing.tsp.types import Observation, State
import chex
from chex import Array
from typing import Any, Callable
import matplotlib.pyplot as plt

class MisActor(nn.Module):
	problem_size: int
	scale: float = 1.0

	def setup(self):
		self.heatmap = self.param('heatmap',
			nn.initializers.constant(value=self.scale), # Initialization function
			(self.problem_size)) # Shape of the parameter

	def __call__(self, obs):
		logits = self.heatmap
		logits = jnp.where(obs.action_mask, -jnp.inf, logits)
		return logits

class SparseHeatmapActor(nn.Module):
	problem_size: int
	num_edges: int
	scale: float = 1.0

	def setup(self):
		self.heatmap = self.param('heatmap',
			nn.initializers.constant(value=self.scale), # Initialization function
			(self.problem_size, int(self.num_edges / self.problem_size))) # Shape of the parameter

	def __call__(self, state):
		"""
		1. init logits with -inf with shape [problem_size]
		2. set logits at indices of receivers of the current position to the heatmap value
		3. set logits to -inf at indices of visited nodes
		"""
		k = self.heatmap.shape[1]
		logits = jnp.full((self.problem_size), -jnp.inf)
		# idx = state.graph.receivers[state.graph.senders == state.position]
		# alternative but probably not jit compatible because of the potentially variable shape
		idx = state.graph.receivers.reshape(self.problem_size, k)[state.position]
		logits = logits.at[idx].set(self.heatmap[state.position])
		logits = jnp.where(state.visited_mask == 1, -jnp.inf, logits)
		return logits

class HeatmapActor(nn.Module):
	problem_size: int
	scale: float = 1.0

	def setup(self):
		self.heatmap = self.param('heatmap',
			nn.initializers.constant(value=self.scale), # Initialization function
			(self.problem_size, self.problem_size)) # Shape of the parameter

	@nn.compact
	def __call__(self, state):
		logits = self.heatmap[state.position]
		logits = jnp.where(state.visited_mask == 1, -jnp.inf, logits)
		return logits


def nearest_neighbor(coordinates, position, visited_mask):
	"""Select the nearest neighbor action.
	Args:
	state: a TSP state.
	Returns:
	action: a feasible action.
	"""
	dists = jnp.linalg.norm(coordinates[position] - coordinates, axis=1)
	dists_modified = jax.lax.select(visited_mask, jnp.full(visited_mask.shape, jnp.inf), dists)
	action = jnp.argmin(dists_modified)
	return action

def knn_graph(coordinates: jnp.array, k: int, include_coordinates:bool = False, include_self_loops:bool = False, force_undirected: bool = False) -> jraph.GraphsTuple:
	"""A naive! implementation to compute the k-nearest neighbor graph of a set of points in jax. Does not inlcude self loops
	Args:
	coordinates: a jnp.array of shape [n, d] representing the coordinates of the points.
	k: the number of neighbors to consider.
	include_coordinates: whether to include the coordinates as node features in the graph.
	include_self_loops: whether to include self loops. Is accomplished by setting diagonale to inf so might not work in case k>=n
	force_undirected: whether to force the graph to be undirected. If True, the graph will be forced to be undirected
	by adding the edges in both directions. WARNING doesnt aggregate features in case of both edges (i,j), (j,i) already existing and 
	Returns:
	graph: a jraph.GraphsTuple representing the k-nearest neighbor graph.
	"""
	if force_undirected:
		raise NotImplementedError
	distances = jnp.linalg.norm(coordinates[:, None] - coordinates, axis=-1)
	if not include_self_loops:
		diag_elements = jnp.diag_indices_from(distances)
		distances = distances.at[diag_elements].set(jnp.inf)
	knn = jnp.argpartition(distances, kth=k, axis=-1)[:, :k]
	# alternative to argpartition
	# knn = jnp.argsort(distances, axis=-1)[:, :k]
	# this choice of senders and receivers connects each node to its k nearest neighbors with the edge directed from the node to its neighbors
	# originally I had it the other way around which feels more natural for the gnn
	# as in sending messages from the close nodes to the node in question
	# but this way gives a fixed number of nodes to decide on the next action when deciding where to go from node i next
	# this simplifies the implementation of the actor as jax requires fixed shapes for jit

	receivers = knn.ravel()
	senders = jnp.repeat(jnp.arange(coordinates.shape[0]), k)
	edges = distances[senders, receivers]
	if include_coordinates:
		nodes = coordinates
	else:
		nodes = None
	graph = jraph.GraphsTuple(nodes=nodes, edges=edges, senders=senders, receivers=receivers, globals=None, n_node=jnp.array([coordinates.shape[0]]), n_edge=jnp.array([senders.shape[0]]))
	return graph

def fully_connected_graph(coordinates: jnp.array, include_coordinates: bool = False):
	"""Constructs a fully connected graph as a jraph.GraphsTuple from a set of points in jax. Does not include self loops."""
	distances = jnp.linalg.norm(coordinates[:, None] - coordinates, axis=-1)
	num_nodes = coordinates.shape[0]
	senders = jnp.tile(jnp.arange(0, num_nodes-1), (num_nodes,1))
	receivers = jnp.tile(jnp.arange(0, num_nodes), (num_nodes-1,1)).T.ravel()
	senders = (senders + jax.numpy.triu(jnp.ones_like(senders), k=0)).ravel()
	edges = distances[senders, receivers]
	if include_coordinates:
		nodes = coordinates
	else:
		nodes = None
	graph = jraph.GraphsTuple(nodes=nodes, edges=edges, senders=senders, receivers=receivers, globals=None, n_node=jnp.array([num_nodes]), n_edge=jnp.array([senders.shape[0]]))
	return graph



