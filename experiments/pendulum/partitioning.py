import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection

from neural_barrier_functions.partitioning import Partitioning


def plot_partitioning(partitioning):
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

    rect_init = plt.Rectangle((-0.01, -0.01), 0.02, 0.02, color='g', fill=False)
    ax.add_patch(rect_init)

    rect_safe = plt.Rectangle((-0.261799388, -1.0), 2 * 0.261799388, 2.0, color='r', fill=False)
    ax.add_patch(rect_safe)

    plt.xlim(-1.2 * 0.261799388, 1.2 * 0.261799388)
    plt.ylim(-1.2, 1.2)

    plt.show()


def nndm_partitioning(config, dynamics):
    partitioning_config = config['partitioning']
    assert partitioning_config['method'] == 'grid'

    x1_lower, x2_lower = dynamics._state_space[0]
    x1_upper, x2_upper = dynamics._state_space[1]

    x1_space = torch.linspace(x1_lower, x1_upper, partitioning_config['num_slices'][0] + 1)
    x2_space = torch.linspace(x2_lower, x2_upper, partitioning_config['num_slices'][1] + 1)

    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    x1_cell_widths = (x1_space[1:] - x1_space[:-1]) / 2
    x2_cell_widths = (x2_space[1:] - x2_space[:-1]) / 2

    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)
    cell_widths = torch.cartesian_prod(x1_cell_widths, x2_cell_widths)
    lower_x, upper_x = cell_centers - cell_widths, cell_centers + cell_widths

    initial_mask = dynamics.initial(cell_centers, cell_widths)
    safe_mask = dynamics.safe(cell_centers, cell_widths)
    unsafe_mask = dynamics.unsafe(cell_centers, cell_widths)

    partitioning = Partitioning(
        (lower_x[initial_mask], upper_x[initial_mask]),
        (lower_x[safe_mask], upper_x[safe_mask]),
        (lower_x[unsafe_mask], upper_x[unsafe_mask]),
        (lower_x, upper_x)
    )

    # plot_partitioning(partitioning)

    return partitioning
