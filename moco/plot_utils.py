import matplotlib.pyplot as plt

def plot_tsp(ax, coordinates, tour):
  """Plot a TSP problem and tour.
  Args:
    ax: a matplotlib axes.
    coordinates: a TSP problem.
    tour: tour as the list of indices.
    """
  ax.scatter(coordinates[:, 0], coordinates[:, 1], marker='o')
  for i,j in zip(tour, tour[1:]):
    ax.plot(coordinates[[i,j], 0], coordinates[[i,j], 1], 'k-')
  ax.plot(coordinates[[tour[-1], tour[0]], 0], coordinates[[tour[-1], tour[0]], 1], 'k-')


def plot_tsp_grid(coordinates, tours):
  """Plot a grid of TSP problems and tours.
  Args:
    coordinates: a batch of TSP problems.
    tours: a batch of tours as the list of indices.
    """
  n = len(tours)
  fig, axs = plt.subplots(1, n, figsize=(3*n, 3))
  for i in range(n):
    plot_tsp(axs[i], coordinates[i], tours[i])

  return fig, axs  
