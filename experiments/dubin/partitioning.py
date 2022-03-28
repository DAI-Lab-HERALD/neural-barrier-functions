import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection

from learned_cbf.partitioning import Partitioning


def plot_partitioning(partitioning):
    """
    Plot partitioning, but ignore direction partitioning (i.e. only the partitioning on x and y) since the safe set is
    only defined by these, and the initial set is defined by x and y, and a limited range directions.
    TODO: Visualize the direction of travel for the initial sets
    :param partitioning:
    :return:
    """
    fig, ax = plt.subplots()

    patch_collection = []
    for lower, width in zip(partitioning.initial.lower, partitioning.initial.width):
        rect = plt.Rectangle(lower[..., :2], width[0], width[1])
        patch_collection.append(rect)
    ax.add_collection(PatchCollection(patch_collection, color='g', alpha=0.1 / 45, linewidth=1))

    patch_collection = []
    for lower, width in zip(partitioning.safe.lower, partitioning.safe.width):
        rect = plt.Rectangle(lower[..., :2], width[0], width[1])
        patch_collection.append(rect)
    ax.add_collection(PatchCollection(patch_collection, color='b', alpha=0.1 / 45, linewidth=1))

    patch_collection = []
    for lower, width in zip(partitioning.unsafe.lower, partitioning.unsafe.width):
        rect = plt.Rectangle(lower[..., :2], width[0], width[1])
        patch_collection.append(rect)
    ax.add_collection(PatchCollection(patch_collection, color='r', alpha=0.1 / 45, linewidth=1))

    rect_init = plt.Rectangle((-0.1, -2.0), 0.2, 0.2, color='g', fill=False)
    ax.add_patch(rect_init)

    circle_unsafe = plt.Circle((0.0, 0.0), math.sqrt(0.04), color='r', fill=False)
    ax.add_patch(circle_unsafe)

    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)

    plt.show()


def dubins_car_partitioning(config, dynamics):
    partitioning_config = config['partitioning']

    assert partitioning_config['method'] == 'grid'

    x1_space = torch.linspace(-2.0, 2.0, partitioning_config['num_slices'][0] + 1)
    x1_cell_width = (x1_space[1] - x1_space[0]) / 2
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2

    x2_space = torch.linspace(-2.0, 2.0, partitioning_config['num_slices'][1] + 1)
    x2_cell_width = (x2_space[1] - x2_space[0]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    x3_space = torch.linspace(-np.pi / 2, np.pi / 2, partitioning_config['num_slices'][2] + 1)
    x3_cell_width = (x3_space[1] - x3_space[0]) / 2
    x3_slice_centers = (x3_space[:-1] + x3_space[1:]) / 2

    cell_width = torch.stack([x1_cell_width, x2_cell_width, x3_cell_width], dim=-1)
    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers, x3_slice_centers)
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

    plot_partitioning(partitioning)

    return partitioning
