import json
import math

import matplotlib
import numpy as np
import torch
from bound_propagation import HyperRectangle, IntervalBounds, LinearBounds
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from tqdm import tqdm

from learned_cbf.bounds import bounds, LearnedCBFBoundModelFactory
from learned_cbf.discretization import ButcherTableau, BoundButcherTableau

from .dynamics import PolynomialUpdate, BoundPolynomialUpdate, NominalPolynomialUpdate, BoundNominalPolynomialUpdate


def bound_propagation(model, lower_x, upper_x, config):
    input_bounds = HyperRectangle(lower_x, upper_x)

    factory = LearnedCBFBoundModelFactory()
    factory.register(PolynomialUpdate, BoundPolynomialUpdate)
    factory.register(NominalPolynomialUpdate, BoundNominalPolynomialUpdate)
    factory.register(ButcherTableau, BoundButcherTableau)
    model = factory.build(model)

    ibp_bounds = model.ibp(input_bounds).cpu()
    crown_bounds = model.crown(input_bounds).cpu()

    input_bounds = input_bounds.cpu()

    return input_bounds, ibp_bounds, crown_bounds


def plot_partition(model, args, input_bounds, ibp_bounds, crown_bounds, initial, safe, unsafe):
    x1, x2 = input_bounds.lower, input_bounds.upper

    # plt.clf()
    plt.figure(figsize=(2 * 6.4, 2 * 4.8))
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

    surf = ax.plot_surface(x1, x2, y1, color=sns.color_palette('muted')[0], label='CROWN interval', alpha=0.8)
    surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_surface(x1, x2, y2, color=sns.color_palette('muted')[0], alpha=0.8)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot LBP linear bounds
    y_lower = crown_bounds.lower[0][0, 0] * x1 + crown_bounds.lower[0][0, 1] * x2 + crown_bounds.lower[1]
    y_upper = crown_bounds.upper[0][0, 0] * x1 + crown_bounds.upper[0][0, 1] * x2 + crown_bounds.upper[1]

    surf = ax.plot_surface(x1, x2, y_lower, color=sns.color_palette('muted')[2], label='CROWN linear', alpha=0.8, shade=False)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_surface(x1, x2, y_upper, color=sns.color_palette('muted')[2], alpha=0.8, shade=False)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot function
    x1, x2 = input_bounds.lower, input_bounds.upper
    x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
    X = torch.cat(tuple(torch.dstack([x1, x2]))).to(args.device)
    y = model(X).view(50, 50)
    y = y.cpu()

    surf = ax.plot_surface(x1, x2, y, color=sns.color_palette('muted')[3], label='Barrier function', shade=False, alpha=1.0)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    ax.set_xlim(input_bounds.lower[0], input_bounds.upper[0])
    ax.set_ylim(input_bounds.lower[1], input_bounds.upper[1])

    # General plot config
    ax.set_xlabel('$x$', labelpad=12.0)
    ax.set_ylabel('$y$', labelpad=20.0)
    ax.set_zlabel('$B(x, y)$', labelpad=20.0)

    plt.title(f'Bound propagation')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig('figures/polynomial_partition.pdf', bbox_inches='tight')
    # plt.show()


def plot_barrier(model, args, config):
    # Plot function over entire space
    plt.figure(figsize=(2 * 6.4, 2 * 4.8))
    ax = plt.axes(projection='3d')

    x1_space = torch.linspace(-3.5, 2.0, 500)
    x2_space = torch.linspace(-2.0, 1.0, 500)
    x1, x2 = torch.meshgrid(x1_space, x2_space)

    X = torch.cat(tuple(torch.dstack([x1, x2]))).to(args.device)
    y = model(X).view(500, 500)
    y = y.cpu()

    surf = ax.plot_surface(x1, x2, y, cmap=sns.color_palette("rocket", as_cmap=True), alpha=0.8)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    z_plane = 0.0

    circle_init = plt.Circle((1.5, 0), math.sqrt(0.25), color=sns.color_palette('deep')[2], fill=True, alpha=0.8,
                             label='Initial set')
    polygon_init = plt.Polygon(
        np.array([[-1.2, 0.1], [-1.8, 0.1], [-1.8, -0.1], [-1.4, -0.1], [-1.4, -0.5], [-1.2, -0.5]]),
        color=sns.color_palette('deep')[2], fill=True, alpha=0.8)
    ax.add_patch(circle_init)
    ax.add_patch(polygon_init)
    art3d.pathpatch_2d_to_3d(circle_init, z=z_plane, zdir='z')
    art3d.pathpatch_2d_to_3d(polygon_init, z=z_plane, zdir='z')

    circle_unsafe = plt.Circle((-1.0, -1.0), math.sqrt(0.16), color=sns.color_palette('deep')[3], fill=True, alpha=0.8,
                               label='Unsafe set')
    polygon_unsafe = plt.Polygon(np.array([[0.4, 0.1], [0.8, 0.1], [0.8, 0.3], [0.6, 0.3], [0.6, 0.5], [0.4, 0.5]]),
                                 color=sns.color_palette('deep')[3], fill=True, alpha=0.8)
    ax.add_patch(circle_unsafe)
    ax.add_patch(polygon_unsafe)
    art3d.pathpatch_2d_to_3d(circle_unsafe, z=z_plane, zdir='z')
    art3d.pathpatch_2d_to_3d(polygon_unsafe, z=z_plane, zdir='z')

    ax.set_xlim(-3.5, 2.0)
    ax.set_ylim(-2.0, 1.0)

    # General plot config
    ax.set_xlabel('$x$', labelpad=12.0)
    ax.set_ylabel('$y$', labelpad=20.0)
    ax.set_zlabel('$B(x, y)$', labelpad=12.0)

    plt.legend()
    plt.title(f'Barrier function & initial and unsafe sets')
    plt.savefig('figures/barrier_3d.pdf', bbox_inches='tight')
    # plt.show()


def partitions(model, num_slices, args, config, input_bounds_only=False):
    x1_space = torch.linspace(-3.5, 2.0, num_slices + 1, device=args.device)
    x1_cell_width = (x1_space[1] - x1_space[0]) / 2
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2

    x2_space = torch.linspace(-2.0, 1.0, num_slices + 1, device=args.device)
    x2_cell_width = (x2_space[1] - x2_space[0]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    cell_width = torch.stack([x1_cell_width, x2_cell_width], dim=-1)

    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    if input_bounds_only:
        return HyperRectangle(lower_x, upper_x)

    input_bounds, ibp_bounds, crown_bounds = None, None, None
    batch_size = 5000

    for batch_lower_x, batch_upper_x in zip(lower_x.split(batch_size), upper_x.split(batch_size)):
        batch_input_bounds, batch_ibp_bounds, batch_crown_bounds = bound_propagation(model, batch_lower_x, batch_upper_x, config)

        if input_bounds is None:
            input_bounds = batch_input_bounds
            ibp_bounds = batch_ibp_bounds
            crown_bounds = batch_crown_bounds
        else:
            input_bounds = HyperRectangle(torch.cat([input_bounds.lower, batch_input_bounds.lower]), torch.cat([input_bounds.upper, batch_input_bounds.upper]))
            ibp_bounds = IntervalBounds(input_bounds, torch.cat([ibp_bounds.lower, batch_ibp_bounds.lower]), torch.cat([ibp_bounds.upper, batch_ibp_bounds.upper]))
            crown_bounds = LinearBounds(input_bounds,
                                        (torch.cat([crown_bounds.lower[0], batch_crown_bounds.lower[0]]), torch.cat([crown_bounds.lower[1], batch_crown_bounds.lower[1]])),
                                        (torch.cat([crown_bounds.upper[0], batch_crown_bounds.upper[0]]), torch.cat([crown_bounds.upper[1], batch_crown_bounds.upper[1]])))

    return input_bounds, ibp_bounds, crown_bounds


def plot_partitions(model, dynamics, args, config):
    num_slices = 40

    input_bounds, ibp_bounds, crown_bounds = partitions(model, num_slices, args, config)
    crown_interval_bounds = crown_bounds.concretize()

    initial_mask = dynamics.initial(input_bounds.center, input_bounds.width / 2)
    safe_mask = dynamics.safe(input_bounds.center, input_bounds.width / 2)
    unsafe_mask = dynamics.unsafe(input_bounds.center, input_bounds.width / 2)

    max_gap = (crown_interval_bounds.upper - crown_interval_bounds.lower).max()

    for i, initial, safe, unsafe in tqdm(zip(range(len(input_bounds)), initial_mask, safe_mask, unsafe_mask)):
        partition_rect = input_bounds[i]
        partition_ibp = ibp_bounds[i]
        partition_crown = crown_bounds[i]
        interval_crown = crown_interval_bounds[i]

        # if lower < 0 or unsafe and lower < 1:
        # if interval_crown.lower < partition_ibp.lower or interval_crown.upper > partition_ibp.upper:
        #     print((interval_crown.lower, interval_crown.upper), (partition_ibp.lower, partition_ibp.upper))
        #     print(initial, safe, unsafe)

        if i == 657 and interval_crown.upper - interval_crown.lower > max_gap * 0.8:
            print(i)
            plot_partition(model, args, partition_rect, partition_ibp, partition_crown, initial, safe, unsafe)


def plot_heatmaps(model, dynamics, args, config):
    num_slices = 320

    input_bounds, ibp_bounds, crown_bounds = partitions(model, num_slices, args, config)
    crown_interval_bounds = crown_bounds.concretize()

    lower = crown_interval_bounds.lower.view(num_slices, num_slices).transpose(0, 1)
    upper = crown_interval_bounds.upper.view(num_slices, num_slices).transpose(0, 1)

    vmin, vmax = crown_interval_bounds.lower.min(), crown_interval_bounds.upper.max()

    matplotlib.rc('axes', titlepad=20)

    plt.figure(figsize=(2 * 5.6, 2 * 4.8))
    im = plt.imshow(lower, cmap=sns.color_palette("rocket", as_cmap=True), vmin=vmin, vmax=vmax, origin='lower',
               extent=[-3.5, 2.0, -2.0, 1.0], aspect='auto')
    plt.title('Lower bound of $B(x, y)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.colorbar(im)
    # plt.show()
    plt.savefig('figures/heatmap_lower.pdf', bbox_inches='tight')

    plt.figure(figsize=(2 * 5.6, 2 * 4.8))
    im = plt.imshow(upper, cmap=sns.color_palette("rocket", as_cmap=True), vmin=vmin, vmax=vmax, origin='lower',
               extent=[-3.5, 2.0, -2.0, 1.0], aspect='auto')
    plt.title('Upper bound of $B(x, y)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.colorbar(im)
    # plt.show()
    plt.savefig('figures/heatmap_upper.pdf', bbox_inches='tight')

    plt.figure(figsize=(2 * 5.6, 2 * 4.8))
    im = plt.imshow(upper - lower, cmap=sns.color_palette("rocket", as_cmap=True), origin='lower',
               extent=[-3.5, 2.0, -2.0, 1.0], aspect='auto')
    plt.title('Gap between upper and lower bound')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.colorbar(im)
    # plt.show()
    plt.savefig('figures/heatmap_diff.pdf', bbox_inches='tight')


def plot_dynamics(model, dynamics, args, config):
    fig, ax = plt.subplots(figsize=(1.9 * 5.4, 1.9 * 4.8))

    circle_init = plt.Circle((1.5, 0), math.sqrt(0.25), facecolor=(*sns.color_palette('deep')[2], 0.4), edgecolor=(*sns.color_palette('deep')[2], 1.0), fill=True, linewidth=2, label='Initial set')
    polygon_init = plt.Polygon(np.array([[-1.2, 0.1], [-1.8, 0.1], [-1.8, -0.1], [-1.4, -0.1], [-1.4, -0.5], [-1.2, -0.5]]),
                               facecolor=(*sns.color_palette('deep')[2], 0.4), edgecolor=(*sns.color_palette('deep')[2], 1.0), fill=True, linewidth=2)
    ax.add_patch(circle_init)
    ax.add_patch(polygon_init)

    circle_unsafe = plt.Circle((-1.0, -1.0), math.sqrt(0.16), facecolor=(*sns.color_palette('deep')[3], 0.4), edgecolor=(*sns.color_palette('deep')[3], 1.0), fill=True, linewidth=2, label='Unsafe set')
    polygon_unsafe = plt.Polygon(np.array([[0.4, 0.1], [0.8, 0.1], [0.8, 0.3], [0.6, 0.3], [0.6, 0.5], [0.4, 0.5]]),
                                 facecolor=(*sns.color_palette('deep')[3], 0.4), edgecolor=(*sns.color_palette('deep')[3], 1.0), fill=True, linewidth=2)
    ax.add_patch(circle_unsafe)
    ax.add_patch(polygon_unsafe)

    num_slices = 40
    input = partitions(model, num_slices, args, config, input_bounds_only=True).center
    nominal_next = dynamics.nominal_system(input)
    diff = nominal_next - input
    input, diff = input.cpu().numpy(), diff.cpu().numpy()

    plt.quiver(input[..., 0], input[..., 1], diff[..., 0], diff[..., 1], angles='xy', color=sns.color_palette('deep')[0])

    plt.xlabel('$x$')
    plt.ylabel('$y$')

    plt.xlim(-3.5, 2.0)
    plt.ylim(-2.0, 1.0)

    matplotlib.rc('axes', titlepad=20)
    plt.title(f'Polynomial system & initial and unsafe sets')

    plt.legend(loc='lower right')
    plt.savefig('figures/polynomial_dynamics.pdf', bbox_inches='tight')
    # plt.show()


def plot_contour(model, args, config, levels, file_path):
    fig, ax = plt.subplots(figsize=(1.9 * 5.4, 1.9 * 4.8))

    circle_init = plt.Circle((1.5, 0), math.sqrt(0.25), facecolor=(*sns.color_palette('deep')[2], 0.4), edgecolor=(*sns.color_palette('deep')[2], 1.0), fill=True, linewidth=2, label='Initial set')
    polygon_init = plt.Polygon(np.array([[-1.2, 0.1], [-1.8, 0.1], [-1.8, -0.1], [-1.4, -0.1], [-1.4, -0.5], [-1.2, -0.5]]),
                               facecolor=(*sns.color_palette('deep')[2], 0.4), edgecolor=(*sns.color_palette('deep')[2], 1.0), fill=True, linewidth=2)
    ax.add_patch(circle_init)
    ax.add_patch(polygon_init)

    circle_unsafe = plt.Circle((-1.0, -1.0), math.sqrt(0.16), facecolor=(*sns.color_palette('deep')[3], 0.4), edgecolor=(*sns.color_palette('deep')[3], 1.0), fill=True, linewidth=2, label='Unsafe set')
    polygon_unsafe = plt.Polygon(np.array([[0.4, 0.1], [0.8, 0.1], [0.8, 0.3], [0.6, 0.3], [0.6, 0.5], [0.4, 0.5]]),
                                 facecolor=(*sns.color_palette('deep')[3], 0.4), edgecolor=(*sns.color_palette('deep')[3], 1.0), fill=True, linewidth=2)
    ax.add_patch(circle_unsafe)
    ax.add_patch(polygon_unsafe)

    num_points = 200
    x1_space = torch.linspace(-3.5, 2.0, num_points)
    x2_space = torch.linspace(-2.0, 1.0, num_points)

    input = torch.cartesian_prod(x1_space, x2_space)
    z = model(input).view(num_points, num_points).numpy()
    input = input.view(num_points, num_points, -1).numpy()
    x, y = input[..., 0], input[..., 1]

    contour = ax.contour(x, y, z, levels, cmap=sns.color_palette('crest', as_cmap=True), vmin=0.0, linewidths=2)
    ax.clabel(contour, contour.levels, inline=True, fontsize=30)

    plt.xlabel('$x$')
    plt.ylabel('$y$')

    plt.xlim(-3.5, 2.0)
    plt.ylim(-2.0, 1.0)

    matplotlib.rc('axes', titlepad=20)
    plt.title(f'Barrier levelsets for polynomial system')

    plt.legend(loc='lower right')
    plt.savefig(file_path, bbox_inches='tight')
    # plt.show()


def plot_contours(model, args, config):

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 30}

    matplotlib.rc('font', **font)

    levels = [1, 1.5, 2.0, 2.2, 2.4, 2.6]
    file_path = 'figures/polynomial_contour_sbf.pdf'
    plot_contour(model, args, config, levels, file_path)

    with open('models/polynomial_system_sos_barrier.json', 'r') as f:
        sos_barrier_parameters = json.load(f)

    def sos_barrier(x):
        x1, x2 = x[..., 0], x[..., 1]

        agg = 0

        for term in sos_barrier_parameters:
            agg = agg + term['coefficient'] * (x1 ** term['exponents'][0]) * (x2 ** term['exponents'][1])

        return agg

    levels = [1, 5, 10, 20, 50]
    file_path = 'figures/polynomial_contour_sos.pdf'
    plot_contour(sos_barrier, args, config, levels, file_path)


@torch.no_grad()
def plot_bounds_2d(model, dynamics, args, config):

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 22}

    matplotlib.rc('font', **font)

    # plot_barrier(model, args, config)
    # plot_partitions(model, dynamics, args, config)
    # plot_heatmaps(model, dynamics, args, config)
    # plot_dynamics(model, dynamics, args, config)
    plot_contours(model, args, config)
