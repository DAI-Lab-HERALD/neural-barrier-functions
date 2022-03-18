import math
from typing import Tuple

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

from .partitioning import overlap_circle, overlap_rectangle, overlap_outside_circle, \
    overlap_outside_rectangle

LinearBound = Tuple[torch.Tensor, torch.Tensor]


class LinearBounds:
    lower: LinearBound
    upper: LinearBound

    def __init__(self, affine):
        self.lower = affine[0]
        self.upper = affine[1]

    def __getitem__(self, item):
        affine = (
            (self.lower[0][item], self.lower[1][item]),
            (self.upper[0][item], self.upper[1][item])
        )

        return LinearBounds(affine)


class IntervalBounds:
    lower: torch.Tensor
    upper: torch.Tensor

    def __init__(self, bounds):
        self.lower = bounds[0]
        self.upper = bounds[1]

    def __getitem__(self, item):
        return IntervalBounds((self.lower[item], self.upper[item]))


class HyperRectangles:
    lower: torch.Tensor
    upper: torch.Tensor
    affine: LinearBounds
    interval: IntervalBounds

    def __init__(self, lower, upper, affine, interval):
        self.lower, self.upper = lower, upper

        if isinstance(affine, LinearBounds):
            self.affine = affine
        else:
            self.affine = LinearBounds(affine)

        if isinstance(interval, IntervalBounds):
            self.interval = interval
        else:
            self.interval = IntervalBounds(interval)

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    @property
    def max_dist(self):
        A, b = self.distance

        center, diff = self.center, self.width / 2
        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        A = A.transpose(-1, -2)

        max_dist = center.matmul(A) + diff.matmul(A.abs()) + b.unsqueeze(-2)
        max_dist = max_dist.view(max_dist.size()[:-2])

        return max_dist

    @property
    def distance(self):
        # Distance between bounds for each sample
        return self.affine.upper[0] - self.affine.lower[0], self.affine.upper[1] - self.affine.lower[1]

    @property
    def global_bounds(self):
        diff = self.width / 2

        A, b = self.affine.lower
        lower = A.matmul(self.center) - A.abs().matmul(diff) + b

        A, b = self.affine.upper
        upper = A.matmul(self.center) + A.abs().matmul(diff) + b

        return lower, upper

    def __getitem__(self, item):
        return HyperRectangles(self.lower[item], self.upper[item], self.affine[item], self.interval[item])

    def __len__(self):
        return 0 if self.lower is None else self.lower.size(0)


def bound_propagation(model, lower_x, upper_x):
    lower_ibp, upper_ibp = model.ibp(lower_x, upper_x)
    lower_lbp, upper_lbp = model.crown_ibp_linear(lower_x, upper_x)

    input_bounds = lower_x.cpu(), upper_x.cpu()
    ibp_bounds = lower_ibp.cpu(), upper_ibp.cpu()
    lbp_bounds = (lower_lbp[0].cpu(), lower_lbp[1].cpu()), (upper_lbp[0].cpu(), upper_lbp[1].cpu())

    return HyperRectangles(*input_bounds, lbp_bounds, ibp_bounds)


def plot_partition(model, args, rect, initial, safe, unsafe):
    x1, x2 = rect.lower, rect.upper

    plt.clf()
    ax = plt.axes(projection='3d')

    x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 10), torch.linspace(x1[1], x2[1], 10))

    # Plot IBP
    y1, y2 = rect.interval.lower.item(), rect.interval.upper.item()
    y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

    surf = ax.plot_surface(x1, x2, y1, color='yellow', label='IBP', alpha=0.4)
    surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_surface(x1, x2, y2, color='yellow', alpha=0.4)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot LBP interval bounds
    y1, y2 = rect.global_bounds[0].item(), rect.global_bounds[1].item()
    y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

    surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
    surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot LBP linear bounds
    y_lower = rect.affine.lower[0][0, 0] * x1 + rect.affine.lower[0][0, 1] * x2 + rect.affine.lower[1]
    y_upper = rect.affine.upper[0][0, 0] * x1 + rect.affine.upper[0][0, 1] * x2 + rect.affine.upper[1]

    surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Plot function
    x1, x2 = rect.lower, rect.upper
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

    x1_space = torch.linspace(-3.5, 2.0, num_slices + 1, device=args.device)
    x1_cell_width = (x1_space[1] - x1_space[0]) / 2
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2

    x2_space = torch.linspace(-2.0, 1.0, num_slices + 1, device=args.device)
    x2_cell_width = (x2_space[1] - x2_space[0]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    cell_width = torch.stack([x1_cell_width, x2_cell_width], dim=-1)

    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    partitions = bound_propagation(model, lower_x, upper_x)

    # Plot function over entire space
    plt.clf()
    ax = plt.axes(projection='3d')

    x1_space = torch.linspace(-3.5, 2.0, 500, device=args.device)
    x2_space = torch.linspace(-2.0, 1.0, 500, device=args.device)
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

    for partition, initial, safe, unsafe in zip(partitions, initial_mask, safe_mask, unsafe_mask):
        verts = [
            (partition.lower[0], partition.lower[1], y_grid),
            (partition.lower[0], partition.upper[1], y_grid),
            (partition.upper[0], partition.upper[1], y_grid),
            (partition.upper[0], partition.lower[1], y_grid),
            (partition.lower[0], partition.lower[1], y_grid)
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

    for partition, initial, safe, unsafe in tqdm(zip(partitions, initial_mask, safe_mask, unsafe_mask)):
        lower, upper = partition.global_bounds
        # if lower < 0 or unsafe and lower < 1:
        if lower < partition.interval.lower or upper > partition.interval.upper:
            print((lower, upper), (partition.interval.lower, partition.interval.upper))
            print(initial, safe, unsafe)
            plot_partition(model, args, partition, initial, safe, unsafe)
