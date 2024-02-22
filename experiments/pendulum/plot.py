import json
import math
from typing import Tuple

import matplotlib
import torch
from bound_propagation import HyperRectangle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from tqdm import tqdm

from neural_barrier_functions.bounds import NBFBoundModelFactory


def bound_propagation(model, lower_x, upper_x):
    input_bounds = HyperRectangle(lower_x, upper_x)

    ibp_bounds = model.ibp(input_bounds).cpu()
    crown_bounds = model.crown_ibp(input_bounds).cpu()

    input_bounds = input_bounds.cpu()

    return input_bounds, ibp_bounds, crown_bounds


def plot_partition(model, args, input_bounds, ibp_bounds, crown_bounds, initial, safe, unsafe):
    x1, x2 = input_bounds.lower, input_bounds.upper

    plt.clf()
    ax = plt.axes(projection='3d')

    x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 10), torch.linspace(x1[1], x2[1], 10))

    # Plot IBP
    y1, y2 = ibp_bounds.lower.item(), ibp_bounds.upper.item()
    y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

    surf = ax.plot_surface(x1, x2, y1, color='yellow', label='IBP', alpha=0.4)
    surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_surface(x1, x2, y2, color='yellow', alpha=0.4)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot LBP interval bounds
    crown_interval = crown_bounds.concretize()
    y1, y2 = crown_interval.lower.item(), crown_interval.upper.item()
    y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

    surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
    surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot LBP linear bounds
    y_lower = crown_bounds.lower[0][0, 0] * x1 + crown_bounds.lower[0][0, 1] * x2 + crown_bounds.lower[1]
    y_upper = crown_bounds.upper[0][0, 0] * x1 + crown_bounds.upper[0][0, 1] * x2 + crown_bounds.upper[1]

    surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot function
    x1, x2 = input_bounds.lower, input_bounds.upper
    x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
    X = torch.cat(tuple(torch.dstack([x1, x2]))).to(args.device)
    y = model(X).view(50, 50)
    y = y.cpu()

    surf = ax.plot_surface(x1, x2, y, color='red', label='Function to bound', shade=False)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # General plot config
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title(f'Bound propagation')
    plt.legend()

    plt.show()


@torch.no_grad()
def plot_bounds_2d(model, dynamics, args, config):
    num_slices = 80

    factory = NBFBoundModelFactory()
    model = factory.build(model)

    if config['dynamics']['safe_set'] == 'circle':
        x_space = torch.linspace(-3.0, 3.0, num_slices + 1, device=args.device)
    elif config['dynamics']['safe_set'] == 'stripe':
        x_space = torch.linspace(0.0, 8.0, num_slices + 1, device=args.device)
    else:
        raise ValueError('Invalid safe set for population')

    cell_width = (x_space[1] - x_space[0]) / 2
    slice_centers = (x_space[:-1] + x_space[1:]) / 2

    cell_centers = torch.cartesian_prod(slice_centers, slice_centers)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    input_bounds, ibp_bounds, crown_bounds = bound_propagation(model, lower_x, upper_x)

    # Plot function over entire space
    plt.clf()
    ax = plt.axes(projection='3d')

    if config['dynamics']['safe_set'] == 'circle':
        x_space = torch.linspace(-3.0, 3.0, 500)
        x1, x2 = torch.meshgrid(x_space, x_space)
    elif config['dynamics']['safe_set'] == 'stripe':
        x_space = torch.linspace(0, 8.0, 500)
        x1, x2 = torch.meshgrid(x_space, x_space)
    else:
        raise ValueError('Invalid safe set for population')

    X = torch.cat(tuple(torch.dstack([x1, x2]))).to(args.device)
    y = model(X).view(500, 500)
    y = y.cpu()

    surf = ax.plot_surface(x1, x2, y, color='red', alpha=0.8)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    initial_mask = dynamics.initial(cell_centers, cell_width)
    safe_mask = dynamics.safe(cell_centers, cell_width)
    unsafe_mask = dynamics.unsafe(cell_centers, cell_width)

    y_grid = -0.5
    initial_partitions = []
    safe_partitions = []
    unsafe_partitions = []

    for i, initial, safe, unsafe in zip(range(len(input_bounds)), initial_mask, safe_mask, unsafe_mask):
        partition_rect = input_bounds[i]
        verts = [
            (partition_rect.lower[0], partition_rect.lower[1], y_grid),
            (partition_rect.lower[0], partition_rect.upper[1], y_grid),
            (partition_rect.upper[0], partition_rect.upper[1], y_grid),
            (partition_rect.upper[0], partition_rect.lower[1], y_grid),
            (partition_rect.lower[0], partition_rect.lower[1], y_grid)
        ]

        if initial:
            initial_partitions.append(verts)

        if safe:
            safe_partitions.append(verts)

        if unsafe:
            unsafe_partitions.append(verts)

    ax.add_collection3d(Poly3DCollection(initial_partitions, facecolor='g', alpha=0.5, linewidth=1))
    ax.add_collection3d(Poly3DCollection(safe_partitions, facecolor='b', alpha=0.5, linewidth=1))
    ax.add_collection3d(Poly3DCollection(unsafe_partitions, facecolor='r', alpha=0.5, linewidth=1))

    # General plot config
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title(f'Barrier function & partitioning')
    plt.show()

    for i, initial, safe, unsafe in tqdm(zip(range(len(input_bounds)), initial_mask, safe_mask, unsafe_mask)):
        partition_rect = input_bounds[i]
        partition_ibp = ibp_bounds[i]
        partition_crown = crown_bounds[i]

        interval_crown = partition_crown.concretize()

        # if lower < 0 or unsafe and lower < 1:
        if interval_crown.lower < partition_ibp.lower or interval_crown.upper > partition_ibp.upper:
            print((interval_crown.lower, interval_crown.upper), (partition_ibp.lower, partition_ibp.upper))
            print(initial, safe, unsafe)
            plot_partition(model, args, partition_rect, partition_ibp, partition_crown, initial, safe, unsafe)


@torch.no_grad()
def plot_contour(model, args, config, levels, file_path, vmin=-1.0):
    fig, ax = plt.subplots(figsize=(1.9 * 5.4, 1.9 * 4.8))

    safe_set_type = config['dynamics']['safe_set']
    if safe_set_type == 'circle':
        rect_unsafe = plt.Rectangle((-3, -3), 6, 6, facecolor=(*sns.color_palette('deep')[3], 0.4), edgecolor=(*sns.color_palette('deep')[3], 1.0), fill=True, label='Unsafe set')
        ax.add_patch(rect_unsafe)

        circle_unsafe = plt.Circle((0, 0), 2.0, facecolor='w', edgecolor=(*sns.color_palette('deep')[3], 1.0), fill=True, linewidth=2)
        ax.add_patch(circle_unsafe)

        circle_init = plt.Circle((0, 0), 1.5, facecolor=(*sns.color_palette('deep')[2], 0.4), edgecolor=(*sns.color_palette('deep')[2], 1.0), fill=True, linewidth=2, label='Initial set')
        ax.add_patch(circle_init)

        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
    elif safe_set_type == 'stripe':
        plt.plot([2.5, 2.25], [2.25, 2.5], color=sns.color_palette('deep')[2], linewidth=2, label='Initial set')
        plt.plot([2.25, 2.5], [2.25, 2.25], color=sns.color_palette('deep')[2], linewidth=2)
        plt.plot([2.25, 2.25], [2.25, 2.5], color=sns.color_palette('deep')[2], linewidth=2)

        plt.plot([7.5, 0.5], [0.5, 7.5], color=sns.color_palette('deep')[3], label='Unsafe set')
        plt.plot([0.5, 7.5], [0.5, 0.5], color=sns.color_palette('deep')[3], linewidth=2)
        plt.plot([0.5, 0.5], [0.5, 7.5], color=sns.color_palette('deep')[3], linewidth=2)

        plt.xlim(0.0, 8.0)
        plt.ylim(0.0, 8.0)
    else:
        raise ValueError('Invalid safe set for population')

    num_points = 200
    x_space = torch.linspace(-3.0, 3.0, num_points).to(args.device)

    input = torch.cartesian_prod(x_space, x_space)
    z = model(input).view(num_points, num_points).cpu().numpy()
    input = input.view(num_points, num_points, -1).cpu().numpy()
    x, y = input[..., 0], input[..., 1]

    contour = ax.contour(x, y, z, levels, cmap=sns.color_palette('crest', as_cmap=True), vmin=vmin, linewidths=2)
    ax.clabel(contour, contour.levels, inline=True, fontsize=30)

    plt.xlabel('$x$')
    plt.ylabel('$y$')

    matplotlib.rc('axes', titlepad=20)
    plt.title(f'Barrier levelsets for linear system')

    plt.legend(loc='lower right')
    plt.savefig(file_path, bbox_inches='tight')
    # plt.show()


def plot_contours(model, args, config):
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 30}

    matplotlib.rc('font', **font)

    levels = [1, 2, 3, 4, 5]
    file_path = 'figures/population_contour_sbf.pdf'
    plot_contour(model, args, config, levels, file_path)

    with open('models/population_system_sos_barrier.json', 'r') as f:
        sos_barrier_parameters = json.load(f)

    def sos_barrier(x):
        x1, x2 = x[..., 0], x[..., 1]

        agg = 0

        for term in sos_barrier_parameters:
            agg = agg + term['coefficient'] * (x1 ** term['exponents'][0]) * (x2 ** term['exponents'][1])

        return agg

    levels = [1, 5, 10, 20, 50]
    file_path = 'figures/population_contour_sos.pdf'
    plot_contour(sos_barrier, args, config, levels, file_path, vmin=-10.0)


@torch.no_grad()
def plot_heatmap(model, args, config):
    fig, ax = plt.subplots()

    num_points = 1000
    x1_space = torch.linspace(-1.2 * 0.261799388, 1.2 * 0.261799388, num_points).to(args.device)
    x2_space = torch.linspace(-1.2, 1.2, num_points).to(args.device)

    input = torch.cartesian_prod(x1_space, x2_space)
    z = model(input).view(num_points, num_points).clamp(max=2.0).cpu().numpy()
    input = input.view(num_points, num_points, -1).cpu().numpy()
    x, y = input[..., 0], input[..., 1]

    pos = plt.contourf(x, y, z, 20)
    fig.colorbar(pos)

    rect_init = plt.Rectangle((-0.01, -0.01), 0.02, 0.02, color='g', fill=False)
    ax.add_patch(rect_init)

    rect_safe = plt.Rectangle((-0.261799388, -1.0), 2 * 0.261799388, 2.0, color='r', fill=False)
    ax.add_patch(rect_safe)

    plt.xlim(-1.2 * 0.261799388, 1.2 * 0.261799388)
    plt.ylim(-1.2, 1.2)

    plt.show()


@torch.no_grad()
def plot_nominal_dynamics(nominal_dynamics, args, config):
    fig, ax = plt.subplots()

    num_points = 20
    x1_space = torch.linspace(-1.2 * 0.261799388, 1.2 * 0.261799388, num_points).to(args.device)
    x2_space = torch.linspace(-1.2, 1.2, num_points).to(args.device)

    input = torch.cartesian_prod(x1_space, x2_space)
    z = nominal_dynamics(input).view(num_points, num_points, -1).cpu().numpy()
    input = input.view(num_points, num_points, -1).cpu().numpy()
    x, y = input[..., 0], input[..., 1]
    u, v = z[..., 0], z[..., 1]

    plt.quiver(x, y, u - x, v - y, angles='xy')

    rect_init = plt.Rectangle((-0.01, -0.01), 0.02, 0.02, color='g', fill=False)
    ax.add_patch(rect_init)

    rect_safe = plt.Rectangle((-0.261799388, -1.0), 2 * 0.261799388, 2.0, color='r', fill=False)
    ax.add_patch(rect_safe)

    plt.xlim(-1.2 * 0.261799388, 1.2 * 0.261799388)
    plt.ylim(-1.2, 1.2)

    plt.show()
