import matplotlib.pyplot as plt
import numpy as np


def generate_random_walk():
    # Define parameters for the walk
    dims = 1
    step_n = 10000
    step_set = [-1, 0, 1]
    origin = np.zeros((1, dims))
    # Simulate steps in 1D
    step_shape = (step_n, dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]
    # Plot the path
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = fig.add_subplot(111)
    ax.scatter(np.arange(step_n + 1), path, c='blue', alpha=0.25, s=0.05)
    ax.plot(path, c='blue', alpha=0.5, lw=0.5, ls='-')
    ax.plot(0, start, c='red', marker='+')
    ax.plot(step_n, stop, c='black', marker='o')
    plt.title('1D Random Walk')
    plt.tight_layout(pad=0)
    plt.show()
    # plt.savefig('plots/random_walk_1d.png', dpi=250)
