import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection

from neural_barrier_functions.partitioning import Partitioning


def plot_partitioning(partitioning, safe_set_type):
    fig, ax = plt.subplots()

    patch_collection = []
    for lower, width in zip(partitioning.initial.lower, partitioning.initial.width):
        rect = plt.Rectangle(lower, width[0], width[1])
        patch_collection.append(rect)
    ax.add_collection(PatchCollection(patch_collection, color='g', alpha=0.2, linewidth=1))

    patch_collection = []
    for lower, width in zip(partitioning.safe.lower, partitioning.safe.width):
        rect = plt.Rectangle(lower, width[0], width[1])
        patch_collection.append(rect)
    ax.add_collection(PatchCollection(patch_collection, color='b', alpha=0.1, linewidth=1))

    patch_collection = []
    for lower, width in zip(partitioning.unsafe.lower, partitioning.unsafe.width):
        rect = plt.Rectangle(lower, width[0], width[1])
        patch_collection.append(rect)
    ax.add_collection(PatchCollection(patch_collection, color='r', alpha=0.1, linewidth=1))

    if safe_set_type == 'circle':
        circle_init = plt.Circle((0, 0), 1.5, color='g', fill=False)
        ax.add_patch(circle_init)

        circle_safe = plt.Circle((0, 0), 2.0, color='r', fill=False)
        ax.add_patch(circle_safe)

        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
    elif safe_set_type == 'stripe':
        plt.plot([2.5, 2.25], [2.25, 2.5], color='g')
        plt.plot([2.25, 2.5], [2.25, 2.25], color='g')
        plt.plot([2.25, 2.25], [2.25, 2.5], color='g')

        plt.plot([7.5, 0.5], [0.5, 7.5], color='r')
        plt.plot([0.5, 7.5], [0.5, 0.5], color='r')
        plt.plot([0.5, 0.5], [0.5, 7.5], color='r')

        plt.xlim(0.0, 8.0)
        plt.ylim(0.0, 8.0)
    else:
        raise ValueError('Invalid safe set for population')

    ax.set_aspect('equal')
    plt.show()


def population_partitioning(config, dynamics):
    partitioning_config = config['partitioning']
    safe_set_type = config['dynamics']['safe_set']

    assert partitioning_config['method'] == 'grid'

    if safe_set_type == 'circle':
        x_lower, x_upper = -3.0, 3.0
    elif safe_set_type == 'stripe':
        x_lower, x_upper = 0.0, 8.0
    else:
        raise ValueError('Invalid safe set for population')

    x1_space = torch.linspace(x_lower, x_upper, partitioning_config['num_slices'][0] + 1)
    x2_space = torch.linspace(x_lower, x_upper, partitioning_config['num_slices'][1] + 1)

    cell_width = torch.stack([(x1_space[1] - x1_space[0]) / 2, (x2_space[1] - x2_space[0]) / 2])
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    initial_mask = dynamics.initial(cell_centers, cell_width)
    safe_mask = dynamics.safe(cell_centers, cell_width)
    unsafe_mask = dynamics.unsafe(cell_centers, cell_width)

    partitioning = Partitioning(
        (lower_x[initial_mask], upper_x[initial_mask]),
        (lower_x[safe_mask], upper_x[safe_mask]),
        (lower_x[unsafe_mask], upper_x[unsafe_mask]),
        (lower_x, upper_x)
    )

    # plot_partitioning(partitioning, safe_set_type)

    return partitioning
