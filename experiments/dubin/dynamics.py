import math
from typing import Tuple

import numpy as np
import torch
from bound_propagation import BoundModule, IntervalBounds, LinearBounds, Cat, HyperRectangle
from bound_propagation.activation import assert_bound_order, regimes
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Normal

from euler import Euler
from learned_cbf.dynamics import StochasticDynamics
from learned_cbf.utils import overlap_circle, overlap_rectangle, overlap_outside_circle


class DubinsCarUpdate(nn.Module):
    def __init__(self, dynamics_config):
        super().__init__()

        self.dt = dynamics_config['dt']
        self.velocity = dynamics_config['velocity']

        dist = Normal(0.0, torch.tensor(dynamics_config['sigma']))
        self.register_buffer('z', dist.sample((dynamics_config['num_samples'],)).view(-1, 1))

    def forward(self, x):
        x1 = self.velocity * x[..., 2].sin()
        x2 = self.velocity * x[..., 2].cos()
        # x[..., 3] = u, i.e. the control. We assume it's concatenated on the last dimension
        x3 = x[..., 3] + self.z

        x = torch.stack([x1.unsqueeze(0).expand_as(x3), x2.unsqueeze(0).expand_as(x3), x3], dim=-1)
        return x


# @torch.jit.script
def _delta(W_tilde: torch.Tensor, beta_lower: torch.Tensor, beta_upper: torch.Tensor) -> torch.Tensor:
    return torch.where(W_tilde < 0, beta_lower.unsqueeze(-1), beta_upper.unsqueeze(-1))


# @torch.jit.script
def _lambda(W_tilde: torch.Tensor, alpha_lower: torch.Tensor, alpha_upper: torch.Tensor) -> torch.Tensor:
    return torch.where(W_tilde < 0, alpha_lower.unsqueeze(-1), alpha_upper.unsqueeze(-1))


# @torch.jit.script
def crown_backward_dubin_jit(W_tilde: torch.Tensor, alpha: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    bias = torch.sum(W_tilde * _delta(W_tilde, *beta), dim=-1)
    W_tilde = W_tilde * _lambda(W_tilde, *alpha)

    return W_tilde, bias


class BoundDubinsCarUpdate(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None
        self.bounded = False

    def func(self, x):
        return x**3

    def derivative(self, x):
        return 2 * x**2

    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower[..., 0], preactivation.upper[..., 0]
        n, p, np = regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        lower_act, upper_act = self.func(lower), self.func(upper)
        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self.func(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        ###################
        # Negative regime #
        ###################
        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=n, a=d_prime, x=d, y=d_act)

        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=n, a=slope, x=lower, y=lower_act)

        ###################
        # Positive regime #
        ###################
        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=p, a=d_prime, x=d, y=d_act)

        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=p, a=slope, x=upper, y=upper_act)

        #################
        # Crossing zero #
        #################
        # Upper bound #
        # If tangent to lower is above upper, then take direct slope between lower and upper
        direct_upper = np & (slope <= lower_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct_upper, a=slope, x=upper, y=upper_act)

        # Else use bisection to find upper bound on slope.
        implicit_upper = np & (slope > lower_prime)

        d_lower = 0.5 * (math.sqrt(5) + 1) * upper
        # Slope has to attach to (upper, upper^3)
        add_linear(self.alpha_upper, self.beta_upper, mask=implicit_upper, a=self.derivative(d_lower), x=upper, y=upper_act)

        # Lower bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct_lower = np & (slope <= upper_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct_lower, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit_lower = np & (slope > upper_prime)

        d_upper = 0.5 * (math.sqrt(5) + 1) * lower
        # Slope has to attach to (upper, sigma(upper))
        add_linear(self.alpha_lower, self.beta_lower, mask=implicit_lower, a=self.derivative(d_upper), x=lower, y=lower_act)

    @property
    def need_relaxation(self):
        return not self.bounded

    def set_relaxation(self, linear_bounds):
        interval_bounds = linear_bounds.concretize()
        self.alpha_beta(preactivation=interval_bounds)
        self.bounded = True

    def backward_relaxation(self, region):
        linear_bounds = self.initial_linear_bounds(region, 2)
        return linear_bounds, self

    def clear_relaxation(self):
        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None
        self.bounded = False

    def crown_backward(self, linear_bounds):
        assert self.bounded

        # NOTE: The order of alpha and beta are deliberately reverse - this is not a mistake!
        if linear_bounds.lower is None:
            lower = None
        else:
            alpha = self.alpha_upper, self.alpha_lower
            beta = self.beta_upper, self.beta_lower
            lower = crown_backward_dubin_jit(linear_bounds.lower[0][..., 0], alpha, beta)

            lowerA = torch.stack([torch.ones_like(lower[0]), lower[0]], dim=-1)
            lower_bias = linear_bounds.lower[1] + torch.stack([torch.zeros_like(lower[1]), lower[1]], dim=-1)
            lower = (lowerA, lower_bias)

        if linear_bounds.upper is None:
            upper = None
        else:
            alpha = self.alpha_lower, self.alpha_upper
            beta = self.beta_lower, self.beta_upper
            upper = crown_backward_dubin_jit(linear_bounds.upper[0][..., 0], alpha, beta)

            upperA = torch.stack([torch.ones_like(upper[0]), upper[0]], dim=-1)
            upper_bias = linear_bounds.upper[1] + torch.stack([torch.zeros_like(upper[1]), upper[1]], dim=-1)
            upper = (upperA, upper_bias)

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_x_sin_phi(self, bounds):
        x1_lower = bounds.lower[..., 0] + self.module.dt * self.module.velocity * bounds.lower[..., 2].sin()
        x1_upper = bounds.upper[..., 0] + self.module.dt * self.module.velocity * bounds.upper[..., 2].sin()

        return x1_lower, x1_upper

    def ibp_y_cos_phi(self, bounds):
        x2_lower = bounds.lower[..., 1] + self.module.dt * self.module.velocity * torch.min(bounds.lower[..., 2].cos(), bounds.upper[..., 2].cos())

        across_center = (bounds.lower[..., 2] <= 0.0) & (bounds.upper[..., 2] >= 0.0)
        center_max = across_center * torch.ones_like(bounds.upper[..., 1])
        boundary_max = torch.max(bounds.lower[..., 2].cos(), bounds.upper[..., 2].cos())
        x2_upper = bounds.upper[..., 1] + self.module.dt * self.module.velocity * torch.max(boundary_max, center_max)

        return x2_lower, x2_upper

    def ibp_control(self, bounds):
        # x[..., 3] = u, i.e. the control. We assume it's concatenated on the last dimension
        x3_lower = bounds.lower[..., 2] + self.module.dt * (bounds.lower[..., 3] + self.module.z)
        x3_upper = bounds.upper[..., 2] + self.module.dt * (bounds.upper[..., 3] + self.module.z)

        return x3_lower, x3_upper

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        x1_lower, x1_upper = self.ibp_x_sin_phi(bounds)
        x2_lower, x2_upper = self.ibp_y_cos_phi(bounds)
        x3_lower, x3_upper = self.ibp_control(bounds)

        lower = torch.stack([x1_lower.unsqueeze(0).expand_as(x3_lower), x2_lower.unsqueeze(0).expand_as(x3_lower), x3_lower], dim=-1)
        upper = torch.stack([x1_upper.unsqueeze(0).expand_as(x3_upper), x2_upper.unsqueeze(0).expand_as(x3_upper), x3_upper], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        assert in_size == 4

        return 3


def plot_dubins_car():
    dynamics = DubinsCarUpdate({
        'sigma': 0.1,
        'num_samples': 500,
        'velocity': 1.0
      })
    bound = BoundDubinsCarUpdate(dynamics, None)

    x_space = torch.linspace(-2.0, 2.0, 4)
    x_cell_width = (x_space[1] - x_space[0]) / 2
    x_slice_centers = (x_space[:-1] + x_space[1:]) / 2

    phi_space = torch.linspace(-np.pi / 2, np.pi / 2, 4)
    phi_cell_width = (phi_space[1] - phi_space[0]) / 2
    phi_slice_centers = (phi_space[:-1] + phi_space[1:]) / 2

    u_space = torch.linspace(1.0, 2.0, 2)
    u_cell_width = (u_space[1] - u_space[0]) / 2
    u_slice_centers = (u_space[:-1] + u_space[1:]) / 2

    cell_width = torch.stack([x_cell_width, x_cell_width, phi_cell_width, u_cell_width], dim=-1)
    cell_centers = torch.cartesian_prod(x_slice_centers, x_slice_centers, phi_slice_centers, u_slice_centers)

    ibp_bounds = bound.ibp(HyperRectangle.from_eps(cell_centers, cell_width))
    # crown_bounds = bound.crown_ibp(HyperRectangle.from_eps(cell_centers, cell_width))

    for i in range(len(ibp_bounds)):
        # X
        x1, x2 = ibp_bounds.region.lower[i, [0, 2]], ibp_bounds.region.upper[i, [0, 2]]

        plt.clf()
        ax = plt.axes(projection='3d')

        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 10), torch.linspace(x1[1], x2[1], 10))

        # Plot IBP
        y1, y2 = ibp_bounds.lower[0, i, 0].item(), ibp_bounds.upper[0, i, 0].item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='yellow', label='IBP', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='yellow', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # # Plot LBP interval bounds
        # crown_interval = crown_bounds.concretize()
        # y1, y2 = crown_interval.lower.item(), crown_interval.upper.item()
        # y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)
        #
        # surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
        # surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # # Plot LBP linear bounds
        # y_lower = crown_bounds.lower[0][0, 0] * x1 + crown_bounds.lower[0][0, 1] * x2 + crown_bounds.lower[1]
        # y_upper = crown_bounds.upper[0][0, 0] * x1 + crown_bounds.upper[0][0, 1] * x2 + crown_bounds.upper[1]
        #
        # surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d

        # Plot function
        x1, x2 = ibp_bounds.region.lower[i, [0, 2]], ibp_bounds.region.upper[i, [0, 2]]
        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
        x3, x4 = torch.zeros_like(x1), torch.zeros_like(x1)
        X = torch.cat(tuple(torch.dstack([x1, x3, x2, x4])))
        y = dynamics(X)[0, :, 0].view(50, 50)

        surf = ax.plot_surface(x1, x2, y, color='red', label="Dubin's car - x", shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Bound propagation')
        plt.legend()

        plt.show()

        # Y
        x1, x2 = ibp_bounds.region.lower[i, [1, 2]], ibp_bounds.region.upper[i, [1, 2]]

        plt.clf()
        ax = plt.axes(projection='3d')

        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 10), torch.linspace(x1[1], x2[1], 10))

        # Plot IBP
        y1, y2 = ibp_bounds.lower[0, i, 1].item(), ibp_bounds.upper[0, i, 1].item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='yellow', label='IBP', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='yellow', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # # Plot LBP interval bounds
        # crown_interval = crown_bounds.concretize()
        # y1, y2 = crown_interval.lower.item(), crown_interval.upper.item()
        # y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)
        #
        # surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
        # surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d

        # # Plot LBP linear bounds
        # y_lower = crown_bounds.lower[0][0, 0] * x1 + crown_bounds.lower[0][0, 1] * x2 + crown_bounds.lower[1]
        # y_upper = crown_bounds.upper[0][0, 0] * x1 + crown_bounds.upper[0][0, 1] * x2 + crown_bounds.upper[1]
        #
        # surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d

        # Plot function
        x1, x2 = ibp_bounds.region.lower[i, [1, 2]], ibp_bounds.region.upper[i, [1, 2]]
        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
        x3, x4 = torch.zeros_like(x1), torch.zeros_like(x1)
        X = torch.cat(tuple(torch.dstack([x3, x1, x2, x4])))
        y = dynamics(X)[0, :, 1].view(50, 50)

        surf = ax.plot_surface(x1, x2, y, color='red', label="Dubin's car - y", shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Bound propagation')
        plt.legend()

        plt.show()


class DubinsFixedStrategy(nn.Module):
    def forward(self, x):
        u = -x[..., 2].sin() + 3 * (self.x_sin_phi(x) + self.y_cos_phi(x)) * self.div(x)
        return u.unsqueeze(-1)

    def div(self, x):
        return 1.0 / (0.5 + x[..., 0]**2 + x[..., 1]**2)

    def x_sin_phi(self, x):
        return x[..., 0] * x[..., 2].sin()

    def y_cos_phi(self, x):
        return x[..., 1] * x[..., 2].cos()


class BoundDubinsFixedStrategy(BoundModule):
    @property
    def need_relaxation(self):
        return False

    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

    def crown_backward(self, linear_bounds):
        raise NotImplementedError()

    def ibp_div(self, bounds):
        across_center1 = (bounds.lower[..., 0] <= 0) & (bounds.upper[..., 0] >= 0)
        across_center2 = (bounds.lower[..., 1] <= 0) & (bounds.upper[..., 1] >= 0)

        min1 = torch.min(bounds.lower[..., 0] ** 2, bounds.upper[..., 0] ** 2) * (~across_center1)
        min2 = torch.min(bounds.lower[..., 1] ** 2, bounds.upper[..., 1] ** 2) * (~across_center2)

        max1 = torch.max(bounds.lower[..., 0] ** 2, bounds.upper[..., 0] ** 2)
        max2 = torch.max(bounds.lower[..., 1] ** 2, bounds.upper[..., 1] ** 2)

        lower_div = 1.0 / (0.5 + max1 + max2)
        upper_div = 1.0 / (0.5 + min1 + min2)

        return lower_div, upper_div

    def ibp_x_sin_phi(self, bounds):
        x_sin_phi = torch.stack([
            bounds.lower[..., 0] * bounds.lower[..., 2].sin(),
            bounds.lower[..., 0] * bounds.upper[..., 2].sin(),
            bounds.upper[..., 0] * bounds.lower[..., 2].sin(),
            bounds.upper[..., 0] * bounds.upper[..., 2].sin(),
        ], dim=-1)
        lower_x_sin_phi = torch.min(x_sin_phi, dim=-1).values
        upper_x_sin_phi = torch.max(x_sin_phi, dim=-1).values

        return lower_x_sin_phi, upper_x_sin_phi

    def ibp_y_cos_phi(self, bounds):
        lower_cos = torch.min(bounds.lower[..., 2].cos(), bounds.upper[..., 2].cos())
        across_center3 = (bounds.lower[..., 2] <= 0) & (bounds.upper[..., 2] >= 0)
        upper_cos = torch.max(torch.max(bounds.lower[..., 2].cos(), bounds.upper[..., 2].cos()), torch.ones_like(lower_cos) * across_center3)

        y_cos_phi = torch.stack([
            bounds.lower[..., 0] * lower_cos,
            bounds.lower[..., 0] * upper_cos,
            bounds.upper[..., 0] * lower_cos,
            bounds.upper[..., 0] * upper_cos,
        ], dim=-1)
        lower_y_cos_phi = torch.min(y_cos_phi, dim=-1).values
        upper_y_cos_phi = torch.max(y_cos_phi, dim=-1).values

        return lower_y_cos_phi, upper_y_cos_phi

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        lower = -bounds.upper[..., 2].sin()
        upper = -bounds.lower[..., 2].sin()

        lower_div, upper_div = self.ibp_div(bounds)
        lower_x_sin_phi, upper_x_sin_phi = self.ibp_x_sin_phi(bounds)
        lower_y_cos_phi, upper_y_cos_phi = self.ibp_y_cos_phi(bounds)

        lower_nom = 3 * (lower_x_sin_phi + lower_y_cos_phi)
        upper_nom = 3 * (upper_x_sin_phi + upper_y_cos_phi)

        frac = torch.stack([
            lower_nom * lower_div,
            lower_nom * upper_div,
            upper_nom * lower_div,
            upper_nom * upper_div,
        ], dim=-1)
        lower_frac = torch.min(frac, dim=-1).values
        upper_frac = torch.max(frac, dim=-1).values

        lower += lower_frac
        upper += upper_frac

        return IntervalBounds(bounds.region, lower.unsqueeze(-1), upper.unsqueeze(-1))

    def propagate_size(self, in_size):
        assert in_size == 3

        return 1


def plot_dubins_car_fixed_strategy_div():
    dynamics = DubinsFixedStrategy()
    bound = BoundDubinsFixedStrategy(dynamics, None)

    x_space = torch.linspace(-2.0, 2.0, 4)
    x_cell_width = (x_space[1] - x_space[0]) / 2
    x_slice_centers = (x_space[:-1] + x_space[1:]) / 2

    phi_space = torch.linspace(-np.pi / 2, np.pi / 2, 2)
    phi_cell_width = (phi_space[1] - phi_space[0]) / 2
    phi_slice_centers = (phi_space[:-1] + phi_space[1:]) / 2

    cell_width = torch.stack([x_cell_width, x_cell_width, phi_cell_width], dim=-1)
    cell_centers = torch.cartesian_prod(x_slice_centers, x_slice_centers, phi_slice_centers)

    input_bounds = HyperRectangle.from_eps(cell_centers, cell_width)

    ibp_bounds = bound.ibp_div(input_bounds)
    # crown_bounds = bound.crown_ibp(HyperRectangle.from_eps(cell_centers, cell_width))

    for i in range(len(input_bounds)):
        # X
        x1, x2 = input_bounds.lower[i, [0, 1]], input_bounds.upper[i, [0, 1]]

        plt.clf()
        ax = plt.axes(projection='3d')

        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 10), torch.linspace(x1[1], x2[1], 10))

        # Plot IBP
        y1, y2 = ibp_bounds[0][i].item(), ibp_bounds[1][i].item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='yellow', label='IBP', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='yellow', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # # Plot LBP interval bounds
        # crown_interval = crown_bounds.concretize()
        # y1, y2 = crown_interval.lower.item(), crown_interval.upper.item()
        # y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)
        #
        # surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
        # surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # # Plot LBP linear bounds
        # y_lower = crown_bounds.lower[0][0, 0] * x1 + crown_bounds.lower[0][0, 1] * x2 + crown_bounds.lower[1]
        # y_upper = crown_bounds.upper[0][0, 0] * x1 + crown_bounds.upper[0][0, 1] * x2 + crown_bounds.upper[1]
        #
        # surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d

        # Plot function
        x1, x2 = input_bounds.lower[i, [0, 1]], input_bounds.upper[i, [0, 1]]
        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
        x3 = torch.zeros_like(x1)
        X = torch.cat(tuple(torch.dstack([x1, x2, x3])))
        y = dynamics.div(X).view(50, 50)

        surf = ax.plot_surface(x1, x2, y, color='red', label='Div in strategy', shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Bound propagation')
        plt.legend()

        plt.show()


def plot_dubins_car_fixed_strategy_x_sin_phi():
    dynamics = DubinsFixedStrategy()
    bound = BoundDubinsFixedStrategy(dynamics, None)

    x1_space = torch.linspace(-2.0, 2.0, 4)
    x1_cell_width = (x1_space[1] - x1_space[0]) / 2
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2

    x2_space = torch.linspace(-2.0, 2.0, 2)
    x2_cell_width = (x2_space[1] - x2_space[0]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    phi_space = torch.linspace(-np.pi / 2, np.pi / 2, 4)
    phi_cell_width = (phi_space[1] - phi_space[0]) / 2
    phi_slice_centers = (phi_space[:-1] + phi_space[1:]) / 2

    cell_width = torch.stack([x1_cell_width, x2_cell_width, phi_cell_width], dim=-1)
    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers, phi_slice_centers)

    input_bounds = HyperRectangle.from_eps(cell_centers, cell_width)

    ibp_bounds = bound.ibp_x_sin_phi(input_bounds)
    # crown_bounds = bound.crown_ibp(HyperRectangle.from_eps(cell_centers, cell_width))

    for i in range(len(input_bounds)):
        # X
        x1, x2 = input_bounds.lower[i, [0, 2]], input_bounds.upper[i, [0, 2]]

        plt.clf()
        ax = plt.axes(projection='3d')

        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 10), torch.linspace(x1[1], x2[1], 10))

        # Plot IBP
        y1, y2 = ibp_bounds[0][i].item(), ibp_bounds[1][i].item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='yellow', label='IBP', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='yellow', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # # Plot LBP interval bounds
        # crown_interval = crown_bounds.concretize()
        # y1, y2 = crown_interval.lower.item(), crown_interval.upper.item()
        # y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)
        #
        # surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
        # surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # # Plot LBP linear bounds
        # y_lower = crown_bounds.lower[0][0, 0] * x1 + crown_bounds.lower[0][0, 1] * x2 + crown_bounds.lower[1]
        # y_upper = crown_bounds.upper[0][0, 0] * x1 + crown_bounds.upper[0][0, 1] * x2 + crown_bounds.upper[1]
        #
        # surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d

        # Plot function
        x1, x2 = input_bounds.lower[i, [0, 2]], input_bounds.upper[i, [0, 2]]
        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
        x3 = torch.zeros_like(x1)
        X = torch.cat(tuple(torch.dstack([x1, x3, x2])))
        y = dynamics.x_sin_phi(X).view(50, 50)

        surf = ax.plot_surface(x1, x2, y, color='red', label='Div in strategy', shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Bound propagation')
        plt.legend()

        plt.show()


def plot_dubins_car_fixed_strategy_y_cos_phi():
    dynamics = DubinsFixedStrategy()
    bound = BoundDubinsFixedStrategy(dynamics, None)

    x1_space = torch.linspace(-2.0, 2.0, 2)
    x1_cell_width = (x1_space[1] - x1_space[0]) / 2
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2

    x2_space = torch.linspace(-2.0, 2.0, 4)
    x2_cell_width = (x2_space[1] - x2_space[0]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    phi_space = torch.linspace(-np.pi / 2, np.pi / 2, 4)
    phi_cell_width = (phi_space[1] - phi_space[0]) / 2
    phi_slice_centers = (phi_space[:-1] + phi_space[1:]) / 2

    cell_width = torch.stack([x1_cell_width, x2_cell_width, phi_cell_width], dim=-1)
    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers, phi_slice_centers)

    input_bounds = HyperRectangle.from_eps(cell_centers, cell_width)

    ibp_bounds = bound.ibp_y_cos_phi(input_bounds)
    # crown_bounds = bound.crown_ibp(HyperRectangle.from_eps(cell_centers, cell_width))

    for i in range(len(input_bounds)):
        # X
        x1, x2 = input_bounds.lower[i, [1, 2]], input_bounds.upper[i, [1, 2]]

        plt.clf()
        ax = plt.axes(projection='3d')

        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 10), torch.linspace(x1[1], x2[1], 10))

        # Plot IBP
        y1, y2 = ibp_bounds[0][i].item(), ibp_bounds[1][i].item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='yellow', label='IBP', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='yellow', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # # Plot LBP interval bounds
        # crown_interval = crown_bounds.concretize()
        # y1, y2 = crown_interval.lower.item(), crown_interval.upper.item()
        # y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)
        #
        # surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
        # surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # # Plot LBP linear bounds
        # y_lower = crown_bounds.lower[0][0, 0] * x1 + crown_bounds.lower[0][0, 1] * x2 + crown_bounds.lower[1]
        # y_upper = crown_bounds.upper[0][0, 0] * x1 + crown_bounds.upper[0][0, 1] * x2 + crown_bounds.upper[1]
        #
        # surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d
        #
        # surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d

        # Plot function
        x1, x2 = input_bounds.lower[i, [1, 2]], input_bounds.upper[i, [1, 2]]
        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
        x3 = torch.zeros_like(x1)
        X = torch.cat(tuple(torch.dstack([x3, x1, x2])))
        y = dynamics.y_cos_phi(X).view(50, 50)

        surf = ax.plot_surface(x1, x2, y, color='red', label='Div in strategy', shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Bound propagation')
        plt.legend()

        plt.show()


class DubinsCarNoActuation(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x[..., :1])


class BoundDubinsCarNoActuation(BoundModule):
    @property
    def need_relaxation(self):
        return False

    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

    def crown_backward(self, linear_bounds):
        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (torch.zeros_like(linear_bounds.lower[0][..., :1]), torch.zeros_like(linear_bounds.lower[1]))

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (torch.zeros_like(linear_bounds.upper[0][..., :1]), torch.zeros_like(linear_bounds.upper[1]))

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        return IntervalBounds(bounds.region, torch.zeros_like(bounds.lower[..., :1]), torch.zeros_like(bounds.upper[..., :1]))

    def propagate_size(self, in_size):
        assert in_size == 3

        return 1


class DubinsCarStrategyComposition(Euler, StochasticDynamics):
    def __init__(self, dynamics_config, strategy=None):
        StochasticDynamics.__init__(self, dynamics_config['num_samples'])

        if strategy is None:
            strategy = DubinsCarNoActuation()

        Euler.__init__(self,
            nn.Sequential(
                Cat(strategy),
                DubinsCarUpdate(dynamics_config)
            ),
           dynamics_config['dt']
        )

    def initial(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return overlap_rectangle(lower_x, upper_x,
                                 torch.tensor([-0.1, -2.0, -np.pi / 6.0], device=x.device),
                                 torch.tensor([0.1, -1.8, np.pi / 6.0], device=x.device))

    def safe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return overlap_outside_circle(lower_x[..., :2], upper_x[..., :2], torch.tensor([0.0, 0.0], device=x.device), math.sqrt(0.04))

    def unsafe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return overlap_circle(lower_x[..., :2], upper_x[..., :2], torch.tensor([0.0, 0.0], device=x.device), math.sqrt(0.04))

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return (upper_x[..., 0] >= -2.0) & (lower_x[..., 0] <= 2.0) &\
               (upper_x[..., 1] >= -2.0) & (lower_x[..., 1] <= 2.0) & \
               (upper_x[..., 2] >= -np.pi / 2) & (lower_x[..., 2] <= np.pi / 2)

    @property
    def volume(self):
        return 4.0**2 * np.pi**2


if __name__ == '__main__':
    # plot_dubins_car()
    # plot_dubins_car_fixed_strategy_div()
    # plot_dubins_car_fixed_strategy_x_sin_phi()
    plot_dubins_car_fixed_strategy_y_cos_phi()

    # Given that these four shows true bounds and the rest is simple interval arithmetic, I believe IBP for both the
    # dynamics and the fixed strategy.
