import numpy as np
import numba as nb
import concurrent.futures
from functools import partial

@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16, nb.boolean), nogil=True)
def two_opt_once(distmat, tour, fixed_i = 0, first_accept=True):
    '''in-place operation'''
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i==0 else range(fixed_i, fixed_i+1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i-1], tour[(j+1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (  distmat[node_prev, node_j] 
                        + distmat[node_i, node_next]
                        - distmat[node_prev, node_i] 
                        - distmat[node_j, node_next])                    
            if change < delta:
                p, q, delta = i, j, change
                if first_accept and delta < -1e-6:
                    tour[p: q+1] = np.flip(tour[p: q+1])
                    return delta
                    
    if delta < -1e-6:
        tour[p: q+1] = np.flip(tour[p: q+1])
        return delta
    else:
        return 0.0


@nb.njit(nb.types.Tuple((nb.uint16[:], nb.float32))(nb.float32[:,:], nb.uint16[:], nb.int64, nb.boolean), nogil=True)
def _two_opt_python(distmat, tour, max_iterations=1000, first_accept=True):
    iterations = 0
    tour = tour.copy()
    cumulative_change = 0.0
    min_change = -1.0
    while min_change < -1e-6 and iterations < max_iterations:
        min_change = two_opt_once(distmat, tour, 0, first_accept=first_accept)
        cumulative_change += min_change
        iterations += 1
    return tour, cumulative_change

def batched_two_opt_python(dist: np.ndarray, tours: np.ndarray, max_iterations=1000, first_accept=True):
    dist = dist.astype(np.float32)
    tours = tours.astype(np.uint16)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for tour in tours:
            future = executor.submit(partial(_two_opt_python, distmat=dist, max_iterations=max_iterations, first_accept=first_accept), tour = tour)
            futures.append(future)
        return np.stack([f.result()[0] for f in futures]), np.stack([f.result()[1] for f in futures])

def jax_two_opt_cb(distances, tours, max_iterations=1000, first_accept=True):
    distances = np.array(distances, dtype=np.float32)
    tours = np.array(tours, dtype=np.int32)
    infeasible_tours_mask = tours[:,-1] == -1
    cost_differences = np.zeros(tours.shape[0], dtype=np.float32)
    tours_ = tours[~infeasible_tours_mask]
    tours_ = np.asarray(tours_, dtype=np.uint16)
    max_iterations = int(max_iterations)
    first_accept = bool(first_accept)
    if tours_.shape[0] == 0:
        return tours.astype(np.int32), cost_differences.astype(np.float32)
    else:    
        new_tours, cost_change = batched_two_opt_python(distances, tours_, max_iterations, first_accept)
        cost_differences[~infeasible_tours_mask] = cost_change
        tours[~infeasible_tours_mask] = new_tours
        return tours.astype(np.int32), cost_differences.astype(np.float32)