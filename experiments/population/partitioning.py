import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection

from learned_cbf.partitioning import Partitioning


def plot_partitioning(partitioning, safe_set_type):
    fig, ax = plt.subplots()

    patch_collection = []
    for lower, width in zip(partitioning.initial.lower, partitioning.initial.width):
        rect = plt.Rectangle(lower, width[0], width[1])
        patch_collection.append(rect)
    ax.add_collection(PatchCollection(patch_collection, color='g', alpha=0.1, linewidth=1))

    patch_collection = []
    for lower, width in zip(partitioning.safe.lower, partitioning.safe.width):
        rect = plt.Rectangle(lower, width[0], width[1], color='b', alpha=0.1, linewidth=1)
        patch_collection.append(rect)
    ax.add_collection(PatchCollection(patch_collection, color='b', alpha=0.1, linewidth=1))

    patch_collection = []
    for lower, width in zip(partitioning.unsafe.lower, partitioning.unsafe.width):
        rect = plt.Rectangle(lower, width[0], width[1], color='r', alpha=0.1, linewidth=1)
        patch_collection.append(rect)
    ax.add_collection(PatchCollection(patch_collection, color='r', alpha=0.1, linewidth=1))

    if safe_set_type == 'circle':
        circle_init = plt.Circle((0, 0), 1.0, color='g', fill=False)
        ax.add_patch(circle_init)

        circle_safe = plt.Circle((0, 0), 2.0, color='r', fill=False)
        ax.add_patch(circle_safe)

        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
    elif safe_set_type == 'annulus':
        circle_init1 = plt.Circle((0, 0), 2.0, color='g', fill=False)
        circle_init2 = plt.Circle((0, 0), 2.5, color='g', fill=False)
        ax.add_patch(circle_init1)
        ax.add_patch(circle_init2)

        circle_safe1 = plt.Circle((0, 0), 0.5, color='r', fill=False)
        circle_safe2 = plt.Circle((0, 0), 4.0, color='r', fill=False)
        ax.add_patch(circle_safe1)
        ax.add_patch(circle_safe2)

        plt.xlim(-4.5, 4.5)
        plt.ylim(-4.5, 4.5)
    else:
        raise ValueError('Invalid safe set for population')

    plt.show()


def population_partitioning(config):
    partitioning_config = config['partitioning']
    safe_set_type = config['dynamics']['safe_set']

    assert partitioning_config['method'] == 'grid'

    if safe_set_type == 'circle':
        x_lim = 3.0
    elif safe_set_type == 'annulus':
        x_lim = 4.5
    else:
        raise ValueError('Invalid safe set for population')

    x1_space = torch.linspace(-x_lim, x_lim, partitioning_config['num_slices'][0] + 1)
    x2_space = torch.linspace(-x_lim, x_lim, partitioning_config['num_slices'][1] + 1)

    cell_width = torch.stack([(x1_space[1] - x1_space[0]) / 2, (x2_space[1] - x2_space[0]) / 2])
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    closest_point = torch.min(lower_x.abs(), upper_x.abs())
    farthest_point = torch.max(lower_x.abs(), upper_x.abs())

    if safe_set_type == 'circle':
        initial_mask = closest_point.norm(dim=-1) <= 1.0
        safe_mask = closest_point.norm(dim=-1) <= 2.0
        unsafe_mask = farthest_point.norm(dim=-1) >= 2.0
    elif safe_set_type == 'annulus':
        initial_mask = (farthest_point.norm(dim=-1) >= 2.0) & (closest_point.norm(dim=-1) <= 2.5)
        safe_mask = (farthest_point.norm(dim=-1) >= 0.5) & (closest_point.norm(dim=-1) <= 4.0)
        unsafe_mask = (farthest_point.norm(dim=-1) >= 4.0) | (closest_point.norm(dim=-1) <= 0.5)
    else:
        raise ValueError('Invalid safe set for population')

    partitioning = Partitioning(
        (lower_x[initial_mask], upper_x[initial_mask]),
        (lower_x[safe_mask], upper_x[safe_mask]),
        (lower_x[unsafe_mask], upper_x[unsafe_mask]),
        (lower_x, upper_x)
    )

    # plot_partitioning(partitioning, safe_set_type)

    return partitioning
