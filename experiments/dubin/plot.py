import torch
from bound_propagation import HyperRectangle, BoundModelFactory
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

from bounds import bounds
from learned_cbf.discretization import BoundEuler, Euler, RK4, BoundRK4
from .dynamics import DubinsCarUpdate, BoundDubinsCarUpdate, BoundDubinsFixedStrategy, \
    DubinsFixedStrategy, DubinsCarNoActuation, BoundDubinsCarNoActuation


def bound_propagation(model, lower_x, upper_x, config):
    input_bounds = HyperRectangle(lower_x, upper_x)

    # ibp_bounds = bounds(model, input_bounds, method='ibp', batch_size=config['test']['ibp_batch_size'])
    # crown_bounds = bounds(model, input_bounds, method='crown_ibp_linear', batch_size=config['test']['crown_ibp_batch_size'])

    factory = BoundModelFactory()
    factory.register(DubinsCarUpdate, BoundDubinsCarUpdate)
    factory.register(DubinsFixedStrategy, BoundDubinsFixedStrategy)
    factory.register(DubinsCarNoActuation, BoundDubinsCarNoActuation)
    factory.register(RK4, BoundRK4)
    model = factory.build(model)

    ibp_bounds = model.ibp(input_bounds).cpu()
    crown_bounds = model.crown(input_bounds).cpu()

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

    # factory = BoundModelFactory()
    # factory.register(Polynomial, BoundPolynomial)
    # model = factory.build(model)

    x1_space = torch.linspace(-3.5, 2.0, num_slices + 1, device=args.device)
    x1_cell_width = (x1_space[1] - x1_space[0]) / 2
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2

    x2_space = torch.linspace(-2.0, 1.0, num_slices + 1, device=args.device)
    x2_cell_width = (x2_space[1] - x2_space[0]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    cell_width = torch.stack([x1_cell_width, x2_cell_width], dim=-1)

    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    input_bounds, ibp_bounds, crown_bounds = bound_propagation(model, lower_x, upper_x, config)

    # Plot function over entire space
    plt.clf()
    ax = plt.axes(projection='3d')

    x1_space = torch.linspace(-3.5, 2.0, 500)
    x2_space = torch.linspace(-2.0, 1.0, 500)
    x1, x2 = torch.meshgrid(x1_space, x2_space)

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
