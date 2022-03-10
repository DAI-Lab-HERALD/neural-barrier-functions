import torch
from matplotlib import pyplot as plt

from learned_cbf.partitioning import Partitioning


def plot_partitioning(partitioning):
    fig, ax = plt.subplots()

    for lower, width in zip(partitioning.initial.lower, partitioning.initial.width):
        rect = plt.Rectangle(lower, width[0], width[1], color='g', alpha=0.1, linewidth=1)
        ax.add_patch(rect)

    for lower, width in zip(partitioning.safe.lower, partitioning.safe.width):
        rect = plt.Rectangle(lower, width[0], width[1], color='b', alpha=0.1, linewidth=1)
        ax.add_patch(rect)

    for lower, width in zip(partitioning.unsafe.lower, partitioning.unsafe.width):
        rect = plt.Rectangle(lower, width[0], width[1], color='r', alpha=0.1, linewidth=1)
        ax.add_patch(rect)

    circle_init = plt.Circle((0, 0), 1.0, color='g', fill=False)
    ax.add_patch(circle_init)

    circle_safe = plt.Circle((0, 0), 2.0, color='r', fill=False)
    ax.add_patch(circle_safe)

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()


def population_partitioning():
    x1_space = torch.linspace(-3.0, 3.0, 161)
    x2_space = torch.linspace(-3.0, 3.0, 161)

    cell_width = torch.stack([(x1_space[1] - x1_space[0]) / 2, (x2_space[1] - x2_space[0]) / 2])
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    closest_point = torch.min(lower_x.abs(), upper_x.abs())
    farthest_point = torch.max(lower_x.abs(), upper_x.abs())

    initial_mask = closest_point[:, 0]**2 + closest_point[:, 1]**2 <= 1.0**2
    safe_mask = closest_point[:, 0]**2 + closest_point[:, 1]**2 <= 2.0**2
    unsafe_mask = farthest_point[:, 0]**2 + farthest_point[:, 1]**2 >= 2.0**2

    partitioning = Partitioning(
        (lower_x[initial_mask], upper_x[initial_mask]),
        (lower_x[safe_mask], upper_x[safe_mask]),
        (lower_x[unsafe_mask], upper_x[unsafe_mask]),
        (lower_x, upper_x)
    )

    # plot_partitioning(partitioning)

    return partitioning
