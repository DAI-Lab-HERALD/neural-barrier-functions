import math

import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection

from learned_cbf.partitioning import Partitioning


def plot_partitioning(partitioning):
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

    circle_init = plt.Circle((0, 0), 1.0, color='g', fill=False)
    ax.add_patch(circle_init)

    circle_safe = plt.Circle((0, 0), 2.0, color='r', fill=False)
    ax.add_patch(circle_safe)

    plt.xlim(-3.5, 2.0)
    plt.ylim(-2.0, 1.0)

    plt.show()


def overlap_rectangle(partition_lower, partition_upper, rect_lower, rect_upper):
    # Separating axis theorem
    return partition_upper[..., 0] >= rect_lower[0] and partition_lower[..., 0] <= rect_upper[0] and \
           partition_upper[..., 1] >= rect_lower[1] and partition_lower[..., 1] <= rect_upper[1]


def overlap_circle(partition_lower, partition_upper, center, radius):
    closest_point = torch.max(partition_lower, torch.min(partition_upper, center))
    distance = (closest_point - center).norm(dim=-1)
    return distance <= radius


def overlap_outside_rectangle(partition_lower, partition_upper, rect_lower, rect_upper):
    return partition_upper[..., 0] >= rect_upper[0] or partition_lower[..., 0] <= rect_lower[0] or \
           partition_upper[..., 1] >= rect_upper[1] or partition_lower[..., 1] <= rect_lower[1]


def overlap_outside_circle(partition_lower, partition_upper, center, radius):
    farthest_point = torch.where((partition_lower - center).abs() > (partition_upper - center).abs(), partition_lower, partition_upper)
    distance = (farthest_point - center).norm(dim=-1)
    return distance >= radius


def polynomial_partitioning(config):
    partitioning_config = config['partitioning']

    assert partitioning_config['method'] == 'grid'

    x1_space = torch.linspace(-3.5, 2.0, partitioning_config['num_slices'][0] + 1)
    x1_cell_width = (x1_space[1] - x1_space[0]) / 2
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2

    x2_space = torch.linspace(-2.0, 1.0, partitioning_config['num_slices'][1] + 1)
    x2_cell_width = (x2_space[1] - x2_space[0]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    cell_width = torch.stack([x1_cell_width, x2_cell_width], dim=-1)

    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    initial_mask = overlap_circle(lower_x, upper_x, torch.tensor([1.5, 0]), math.sqrt(0.25)) | \
                   overlap_rectangle(lower_x, upper_x, torch.tensor([-1.8, -0.1]), torch.tensor([-1.2, 0.1])) | \
                   overlap_rectangle(lower_x, upper_x, torch.tensor([-1.4, -0.5]), torch.tensor([-1.2, 0.1]))

    safe_mask = overlap_outside_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0]), math.sqrt(0.16)) & \
                   overlap_outside_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.6, 0.5])) & \
                   overlap_outside_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.8, 0.3]))

    unsafe_mask = overlap_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0]), math.sqrt(0.16)) | \
                   overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.6, 0.5])) | \
                   overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.8, 0.3]))

    partitioning = Partitioning(
        (lower_x[initial_mask], upper_x[initial_mask]),
        (lower_x[safe_mask], upper_x[safe_mask]),
        (lower_x[unsafe_mask], upper_x[unsafe_mask]),
        (lower_x, upper_x)
    )

    plot_partitioning(partitioning)

    return partitioning
