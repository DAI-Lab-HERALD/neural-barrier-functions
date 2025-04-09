import json
import math
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from bound_propagation import BoundModule, IntervalBounds, LinearBounds, Cat, HyperRectangle
from bound_propagation.activation import assert_bound_order, regimes, bisection
from bound_propagation.saturation import Clamp
from matplotlib import pyplot as plt
from torch import nn, distributions
from torch.distributions import Normal

from neural_barrier_functions.bounds import NBFBoundModelFactory
from neural_barrier_functions.discretization import Euler, RK4, Heun, ButcherTableau, BoundButcherTableau
from neural_barrier_functions.dynamics import AdditiveGaussianDynamics
from neural_barrier_functions.utils import overlap_circle, overlap_rectangle, overlap_outside_circle, overlap_outside_rectangle


class DubinsCarUpdate(nn.Module):
    def __init__(self, dynamics_config):
        super().__init__()

        self.velocity = dynamics_config['velocity']

        self.dist = Normal(torch.tensor(dynamics_config['mu']), torch.tensor(dynamics_config['sigma']))

        self.register_buffer('mu', torch.tensor(dynamics_config['mu']))
        self.register_buffer('sigma', torch.tensor(dynamics_config['sigma']))
        self.num_samples = dynamics_config['num_samples']

        dist = Normal(self.mu, self.sigma)
        self.register_buffer('z', dist.sample((self.num_samples, 1)))

    def resample(self):
        dist = Normal(self.mu, self.sigma)
        self.z = dist.sample((self.num_samples, 1)).to(self.z.device)

    def forward(self, x):
        self.resample()

        x1 = self.velocity * x[..., 2].sin()
        x2 = self.velocity * x[..., 2].cos()
        # x[..., 3] = u, i.e. the control. We assume it's concatenated on the last dimension
        x3 = x[..., 3] + z

        if x1.dim() != x3.dim():
            x1 = x1.unsqueeze(0).expand_as(x3)
            x2 = x2.unsqueeze(0).expand_as(x3)

        x = torch.stack([x1, x2, x3, torch.zeros_like(x3)], dim=-1)
        return x


@torch.jit.script
def crown_backward_dubin_jit(W_tilde: torch.Tensor, z: torch.Tensor, alpha: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    _lambda = torch.where(W_tilde[..., :2] < 0, alpha[0].unsqueeze(-2), alpha[1].unsqueeze(-2))
    _delta = torch.where(W_tilde[..., :2] < 0, beta[0].unsqueeze(-2), beta[1].unsqueeze(-2))

    bias = torch.sum(W_tilde[..., :2] * _delta, dim=-1) + z.unsqueeze(-1) * W_tilde[..., 2]
    W_tilde1 = torch.zeros_like(W_tilde[..., 0])
    W_tilde2 = torch.zeros_like(W_tilde[..., 1])
    W_tilde3 = torch.sum(W_tilde[..., :2] * _lambda, dim=-1)
    W_tilde4 = W_tilde[..., 2]

    if W_tilde1.dim() != W_tilde3.dim():
        W_tilde1 = W_tilde1.unsqueeze(0).expand_as(W_tilde3)
        W_tilde2 = W_tilde2.unsqueeze(0).expand_as(W_tilde3)
        W_tilde4 = W_tilde4.unsqueeze(0).expand_as(W_tilde3)

    W_tilde = torch.stack([W_tilde1, W_tilde2, W_tilde3, W_tilde4], dim=-1)

    return W_tilde, bias


class BoundDubinsCarUpdate(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None
        self.bounded = False

    def func(self, x):
        return torch.stack([self.module.velocity * x.sin(), self.module.velocity * x.cos()], dim=-1)

    def derivative(self, x):
        return torch.stack([self.module.velocity * x.cos(), -self.module.velocity * x.sin()], dim=-1)

    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower[..., 2], preactivation.upper[..., 2]
        zero_width, n, p, np = regimes(lower, upper)

        zero = torch.zeros_like(lower).unsqueeze(-1).expand(*lower.size(), 2)

        self.alpha_lower, self.beta_lower = zero.detach().clone(), zero.detach().clone()
        self.alpha_upper, self.beta_upper = zero.detach().clone(), zero.detach().clone()

        lower_act, upper_act = self.func(lower), self.func(upper)
        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self.func(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower).unsqueeze(-1)

        def add_linear(alpha, beta, mask, order, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask, [order]]

            alpha[mask, [order]] = a
            beta[mask, [order]] = y[mask, [order]] - a * x[mask]

        #######################
        # (Almost) zero width #
        #######################
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, torch.min(lower_act[zero_width], upper_act[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, torch.max(lower_act[zero_width], upper_act[zero_width])

        #########################
        # Negative regime - sin #
        #########################
        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=n, order=0, a=slope, x=lower, y=lower_act)

        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=n, order=0, a=d_prime, x=d, y=d_act)

        #########################
        # Negative regime - cos #
        #########################
        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=n, order=1, a=d_prime, x=d, y=d_act)

        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=n, order=1, a=slope, x=lower, y=lower_act)

        #########################
        # Positive regime - sin #
        #########################
        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=p, order=0, a=d_prime, x=d, y=d_act)

        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=p, order=0, a=slope, x=upper, y=upper_act)

        #########################
        # Positive regime - cos #
        #########################
        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=p, order=1, a=d_prime, x=d, y=d_act)

        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=p, order=1, a=slope, x=upper, y=upper_act)

        #######################
        # Crossing zero - sin #
        #######################
        # Upper bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct_upper = np & (slope <= upper_prime)[..., 0]
        add_linear(self.alpha_upper, self.beta_upper, mask=direct_upper, order=0, a=slope, x=upper, y=upper_act)

        # Else use bisection to find upper bound on slope.
        implicit_upper = np & (slope > upper_prime)[..., 0]

        def f_upper(d: torch.Tensor) -> torch.Tensor:
            a_slope = (d.sin() - lower[implicit_upper].sin()) / (d - lower[implicit_upper])
            a_derivative = d.cos()
            return a_slope - a_derivative

        # Bisection will return left and right bounds for d s.t. f_upper(d) is zero
        # Derivative of left bound will over-approximate the slope - hence a true bound
        d_upper, _ = bisection(torch.zeros_like(upper[implicit_upper]), upper[implicit_upper], f_upper)

        # Slope has to attach to (lower, sin(lower))
        add_linear(self.alpha_upper, self.beta_upper, mask=implicit_upper, order=0, a=d_upper.cos(), x=lower, y=lower_act, a_mask=False)

        # Lower bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct_lower = np & (slope <= lower_prime)[..., 0]
        add_linear(self.alpha_lower, self.beta_lower, mask=direct_lower, order=0, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit_lower = np & (slope > lower_prime)[..., 0]

        def f_lower(d: torch.Tensor) -> torch.Tensor:
            a_slope = (upper[implicit_lower].sin() - d.sin()) / (upper[implicit_lower] - d)
            a_derivative = d.cos()
            return a_derivative - a_slope

        # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
        # Derivative of right bound will over-approximate the slope - hence a true bound
        _, d_lower = bisection(lower[implicit_lower], torch.zeros_like(lower[implicit_lower]), f_lower)

        # Slope has to attach to (upper, sin(upper))
        add_linear(self.alpha_lower, self.beta_lower, mask=implicit_lower, order=0, a=d_lower.cos(), x=upper, y=upper_act, a_mask=False)

        #######################
        # Crossing zero - cos #
        #######################
        # Upper bound #
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=np, order=1, a=d_prime, x=d, y=d_act)

        # Lower bound #
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=np, order=1, a=slope, x=lower, y=lower_act)

    @property
    def need_relaxation(self):
        return not self.bounded

    def set_relaxation(self, linear_bounds):
        interval_bounds = linear_bounds.concretize()
        self.alpha_beta(preactivation=interval_bounds)
        self.bounded = True

    def backward_relaxation(self, region):
        linear_bounds = self.initial_linear_bounds(region, 4)
        return linear_bounds, self

    def clear_relaxation(self):
        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None
        self.bounded = False

    def crown_backward(self, linear_bounds, optimize):
        assert self.bounded

        # NOTE: The order of alpha and beta are deliberately reversed - this is not a mistake!
        if linear_bounds.lower is None:
            lower = None
        else:
            alpha = self.alpha_upper, self.alpha_lower
            beta = self.beta_upper, self.beta_lower
            lower = crown_backward_dubin_jit(linear_bounds.lower[0], self.module.z, alpha, beta)

            lower = (lower[0], lower[1] + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            alpha = self.alpha_lower, self.alpha_upper
            beta = self.beta_lower, self.beta_upper
            upper = crown_backward_dubin_jit(linear_bounds.upper[0], self.module.z, alpha, beta)
            upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_sin_phi(self, bounds):
        x1_lower = self.module.velocity * bounds.lower[..., 2].sin()
        x1_upper = self.module.velocity * bounds.upper[..., 2].sin()

        return x1_lower, x1_upper

    def ibp_cos_phi(self, bounds):
        x2_lower = self.module.velocity * torch.min(bounds.lower[..., 2].cos(), bounds.upper[..., 2].cos())

        across_center = (bounds.lower[..., 2] <= 0.0) & (bounds.upper[..., 2] >= 0.0)
        boundary_max = torch.max(bounds.lower[..., 2].cos(), bounds.upper[..., 2].cos())
        x2_upper = self.module.velocity * torch.max(boundary_max, across_center)

        return x2_lower, x2_upper

    def ibp_control(self, bounds):
        # x[..., 3] = u, i.e. the control. We assume it's concatenated on the last dimension
        x3_lower = bounds.lower[..., 3] + self.module.z
        x3_upper = bounds.upper[..., 3] + self.module.z

        return x3_lower, x3_upper

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        x1_lower, x1_upper = self.ibp_sin_phi(bounds)
        x2_lower, x2_upper = self.ibp_cos_phi(bounds)
        x3_lower, x3_upper = self.ibp_control(bounds)

        if x1_lower.dim() != x3_lower.dim():
            x1_lower = x1_lower.unsqueeze(0).expand_as(x3_lower)
            x1_upper = x1_upper.unsqueeze(0).expand_as(x3_upper)
            x2_lower = x2_lower.unsqueeze(0).expand_as(x3_lower)
            x2_upper = x2_upper.unsqueeze(0).expand_as(x3_upper)

        lower = torch.stack([x1_lower, x2_lower, x3_lower, torch.zeros_like(x3_lower)], dim=-1)
        upper = torch.stack([x1_upper, x2_upper, x3_upper, torch.zeros_like(x3_upper)], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        assert in_size == 4

        return 4


class DubinsCarNominalUpdate(nn.Module):
    def __init__(self, dynamics_config):
        super().__init__()

        self.velocity = dynamics_config['velocity']

    def forward(self, x):
        x1 = self.velocity * x[..., 2].sin()
        x2 = self.velocity * x[..., 2].cos()
        # x[..., 3] = u, i.e. the control. We assume it's concatenated on the last dimension
        x3 = x[..., 3]

        x = torch.stack([x1, x2, x3, torch.zeros_like(x3)], dim=-1)
        return x


@torch.jit.script
def crown_backward_dubin_nominal_jit(W_tilde: torch.Tensor, alpha: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    _lambda = torch.where(W_tilde[..., :2] < 0, alpha[0].unsqueeze(-2), alpha[1].unsqueeze(-2))
    _delta = torch.where(W_tilde[..., :2] < 0, beta[0].unsqueeze(-2), beta[1].unsqueeze(-2))

    bias = torch.sum(W_tilde[..., :2] * _delta, dim=-1)
    W_tilde1 = torch.zeros_like(W_tilde[..., 0])
    W_tilde2 = torch.zeros_like(W_tilde[..., 1])
    W_tilde3 = torch.sum(W_tilde[..., :2] * _lambda, dim=-1)
    W_tilde4 = W_tilde[..., 2]
    W_tilde = torch.stack([W_tilde1, W_tilde2, W_tilde3, W_tilde4], dim=-1)

    return W_tilde, bias


class BoundDubinsCarNominalUpdate(BoundDubinsCarUpdate):
    def crown_backward(self, linear_bounds, optimize):
        assert self.bounded

        # NOTE: The order of alpha and beta are deliberately reversed - this is not a mistake!
        if linear_bounds.lower is None:
            lower = None
        else:
            alpha = self.alpha_upper, self.alpha_lower
            beta = self.beta_upper, self.beta_lower
            lower = crown_backward_dubin_nominal_jit(linear_bounds.lower[0], alpha, beta)

            lower = (lower[0], lower[1] + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            alpha = self.alpha_lower, self.alpha_upper
            beta = self.beta_lower, self.beta_upper
            upper = crown_backward_dubin_nominal_jit(linear_bounds.upper[0], alpha, beta)
            upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_control(self, bounds):
        # x[..., 3] = u, i.e. the control. We assume it's concatenated on the last dimension
        x3_lower = bounds.lower[..., 3]
        x3_upper = bounds.upper[..., 3]

        return x3_lower, x3_upper

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        x1_lower, x1_upper = self.ibp_sin_phi(bounds)
        x2_lower, x2_upper = self.ibp_cos_phi(bounds)
        x3_lower, x3_upper = self.ibp_control(bounds)

        lower = torch.stack([x1_lower, x2_lower, x3_lower, torch.zeros_like(x3_lower)], dim=-1)
        upper = torch.stack([x1_upper, x2_upper, x3_upper, torch.zeros_like(x3_upper)], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        assert in_size == 4

        return 4


def plot_dubins_car():
    dynamics = DubinsCarUpdate({
        'mu': 0.1,
        'sigma': 0.1,
        'num_samples': 500,
        'velocity': 1.0
      })
    bound = BoundDubinsCarUpdate(dynamics, None)

    x_space = torch.linspace(-2.0, 2.0, 4)
    x_cell_width = (x_space[1] - x_space[0]) / 2
    x_slice_centers = (x_space[:-1] + x_space[1:]) / 2

    phi_space = torch.linspace(-np.pi / 2, np.pi / 2, 5)
    phi_cell_width = (phi_space[1] - phi_space[0]) / 2
    phi_slice_centers = (phi_space[:-1] + phi_space[1:]) / 2

    u_space = torch.linspace(1.0, 2.0, 2)
    u_cell_width = (u_space[1] - u_space[0]) / 2
    u_slice_centers = (u_space[:-1] + u_space[1:]) / 2

    cell_width = torch.stack([x_cell_width, x_cell_width, phi_cell_width, u_cell_width], dim=-1)
    cell_centers = torch.cartesian_prod(x_slice_centers, x_slice_centers, phi_slice_centers, u_slice_centers)

    ibp_bounds = bound.ibp(HyperRectangle.from_eps(cell_centers, cell_width))
    crown_bounds = bound.crown(HyperRectangle.from_eps(cell_centers, cell_width))
    crown_interval = crown_bounds.concretize()

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

        # Plot LBP interval bounds
        y1, y2 = crown_interval.lower[0, i, 0].item(), crown_interval.upper[0, i, 0].item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Plot LBP linear bounds
        y_lower = crown_bounds.lower[0][i, 0, 0] * x1 + crown_bounds.lower[0][i, 0, 2] * x2 + crown_bounds.lower[1][0, i, 0]
        y_upper = crown_bounds.upper[0][i, 0, 0] * x1 + crown_bounds.upper[0][i, 0, 2] * x2 + crown_bounds.upper[1][0, i, 0]

        surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

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

        # Plot LBP interval bounds
        y1, y2 = crown_interval.lower[0, i, 1].item(), crown_interval.upper[0, i, 1].item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Plot LBP linear bounds
        y_lower = crown_bounds.lower[0][i, 1, 1] * x1 + crown_bounds.lower[0][i, 1, 2] * x2 + crown_bounds.lower[1][0, i, 1]
        y_upper = crown_bounds.upper[0][i, 1, 1] * x1 + crown_bounds.upper[0][i, 1, 2] * x2 + crown_bounds.upper[1][0, i, 1]

        surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

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

    def crown_backward(self, linear_bounds, optimize):
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
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
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


class DubinsCarNoActuation(nn.Linear):
    def __init__(self):
        super().__init__(3, 1, bias=False)

        del self.weight
        self.register_buffer('weight', torch.zeros(1, 3))


class DubinsCarNNStrategy(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            Clamp(min=-1.0, max=1.0)
        )


class DubinSelect(nn.Module):
    def forward(self, x):
        return x[..., :3]


class BoundDubinSelect(BoundModule):
    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (
                torch.cat([linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[0][..., :1])], dim=-1),
                linear_bounds.lower[1]
            )

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (
                torch.cat([linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[0][..., :1])], dim=-1),
                linear_bounds.upper[1]
            )

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        return IntervalBounds(bounds.region, bounds.lower[..., :3], bounds.upper[..., :3])

    def propagate_size(self, in_size):
        assert in_size == 4

        return 3


class DubinsCarStrategyComposition(nn.Sequential, AdditiveGaussianDynamics):
    @property
    def nominal_system(self):
        system = nn.Sequential(
            Cat(self[0].subnetwork),
            Euler(DubinsCarNominalUpdate(self.dynamics_config), self.dynamics_config['dt']),
            DubinSelect()
        )

        return system

    @property
    def v(self):
        return (
            torch.tensor([0.0, 0.0, self.dynamics_config['mu'] * self.dynamics_config['dt']]),
            torch.tensor([0.0, 0.0, self.dynamics_config['sigma'] * self.dynamics_config['dt']])
        )

    def __init__(self, dynamics_config, strategy=None):
        AdditiveGaussianDynamics.__init__(self, dynamics_config['num_samples'])

        if strategy is None:
            strategy = DubinsCarNoActuation()

        nn.Sequential.__init__(self,
            Cat(strategy),
            Euler(DubinsCarUpdate(dynamics_config), dynamics_config['dt']),
            DubinSelect()
        )

        self.initial_set = dynamics_config['initial_set']
        self.unsafe_set = dynamics_config['unsafe_set']
        self.dynamics_config = dynamics_config

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return nn.Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def goal(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return overlap_rectangle(lower_x, upper_x,
                                 torch.tensor([-2.0, 1.8, -np.pi / 2.0], device=x.device),
                                 torch.tensor([2.0, 2.0, np.pi / 2.0], device=x.device))

    def initial(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        if self.initial_set == 'front':
            return overlap_rectangle(lower_x, upper_x,
                                     torch.tensor([-0.1, -2.0, -np.pi / 6.0], device=x.device),
                                     torch.tensor([0.1, -1.8, np.pi / 6.0], device=x.device))
        elif self.initial_set == 'right_dir':
            return overlap_rectangle(lower_x, upper_x,
                                     torch.tensor([-0.1, -2.0, np.pi / 6.0], device=x.device),
                                     torch.tensor([0.1, -1.8, np.pi / 4.0], device=x.device))
        elif self.initial_set == 'left':
            return overlap_rectangle(lower_x, upper_x,
                                     torch.tensor([-0.95, 0.0, 0.0], device=x.device),
                                     torch.tensor([-0.95, 0.0, 0.0], device=x.device))
        else:
            raise ValueError('Invalid initial set')

    def sample_initial(self, num_particles):
        if self.initial_set == 'front':
            dist = distributions.Uniform(torch.tensor([-0.1, -2.0, -np.pi / 6.0]), torch.tensor([0.1, -1.8, np.pi / 6.0]))
        elif self.initial_set == 'right_dir':
            dist = distributions.Uniform(torch.tensor([-0.1, -2.0, np.pi / 6.0]), torch.tensor([0.1, -1.8, np.pi / 4.0]))
        elif self.initial_set == 'left':
            return torch.tensor([[-0.95, 0.0, 0.0]])
        else:
            raise ValueError('Invalid initial set')

        return dist.sample((num_particles,))

    def safe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        if self.unsafe_set == 'barrel':
            return overlap_outside_circle(lower_x[..., :2], upper_x[..., :2], torch.tensor([0.0, 0.0], device=x.device), math.sqrt(0.04))
        elif self.unsafe_set == 'walls':
            return overlap_rectangle(lower_x[..., :2], upper_x[..., :2], torch.tensor([-1.9, -1.9], device=x.device), torch.tensor([1.9, 1.9], device=x.device))
        else:
            raise ValueError('Invalid unsafe set')

    def sample_safe(self, num_particles):
        if self.unsafe_set == 'barrel':
            x = self.sample_state_space(num_particles * 2)
            x = x[self.safe(x)]

            return x[:num_particles]
        elif self.unsafe_set == 'walls':
            dist = distributions.Uniform(torch.tensor([-1.9, -1.9, -np.pi / 2]), torch.tensor([1.9, 1.9, np.pi / 2]))
            return dist.sample((num_particles,))
        else:
            raise ValueError('Invalid unsafe set')

    def unsafe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        if self.unsafe_set == 'barrel':
            return overlap_circle(lower_x[..., :2], upper_x[..., :2], torch.tensor([0.0, 0.0], device=x.device), math.sqrt(0.04))
        elif self.unsafe_set == 'walls':
            return overlap_outside_rectangle(lower_x[..., :2], upper_x[..., :2], torch.tensor([-1.9, -1.9], device=x.device), torch.tensor([1.9, 1.9], device=x.device))
        else:
            raise ValueError('Invalid unsafe set')

    def sample_unsafe(self, num_particles):
        if self.unsafe_set == 'barrel':
            dist = distributions.Uniform(0, 1)
            r = math.sqrt(0.04) * dist.sample((num_particles,)).sqrt()
            theta = dist.sample((num_particles,)) * 2 * np.pi

            dist = distributions.Uniform(-np.pi / 2, np.pi / 2)
            phi = dist.sample((num_particles,))

            return torch.stack([r * theta.cos(), r * theta.sin(), phi], dim=-1)
        elif self.unsafe_set == 'walls':
            rects = [
                ([-2.0, -2.0], [-1.9, -1.9]),
                ([-2.0, -1.9], [-1.9, 1.9]),
                ([-2.0, 1.9], [-1.9, 2.0]),
                ([1.9, -2.0], [2.0, -1.9]),
                ([1.9, -1.9], [2.0, 1.9]),
                ([1.9, 1.9], [2.0, 2.0]),
                ([-1.9, -2.0], [1.9, -1.9]),
                ([-1.9, 1.9], [1.9, 2.0]),
            ]
            areas = [(upper[0] - lower[0]) * (upper[1] - lower[1]) for lower, upper in rects]
            total_area = sum(areas)
            probs = [area / total_area for area in areas]

            dist = distributions.Multinomial(total_count=num_particles, probs=torch.tensor(probs))
            count = dist.sample().int()

            samples = []
            for (lower, upper), n in zip(rects, count):
                dist = distributions.Uniform(torch.tensor(lower), torch.tensor(upper))
                sample = dist.sample((n,))
                samples.append(sample)

            samples = torch.cat(samples)

            dist = distributions.Uniform(-np.pi / 2, np.pi / 2)
            phi = dist.sample((num_particles, 1))

            return torch.cat([samples, phi], dim=-1)
        else:
            raise ValueError('Invalid unsafe set')

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return (upper_x[..., 0] >= -2.0) & (lower_x[..., 0] <= 2.0) &\
               (upper_x[..., 1] >= -2.0) & (lower_x[..., 1] <= 2.0) & \
               (upper_x[..., 2] >= -np.pi / 2) & (lower_x[..., 2] <= np.pi / 2)

    def sample_state_space(self, num_particles):
        dist = torch.distributions.Uniform(torch.tensor([-2.0, -2.0, -np.pi / 2]), torch.tensor([2.0, 2.0, np.pi / 2]))
        return dist.sample((num_particles,))

    @property
    def volume(self):
        return 4.0**2 * np.pi
    

def export_bounds():
    factory = NBFBoundModelFactory()
    factory.register(DubinsCarUpdate, BoundDubinsCarUpdate)
    factory.register(DubinsCarNominalUpdate, BoundDubinsCarNominalUpdate)
    factory.register(DubinsFixedStrategy, BoundDubinsFixedStrategy)
    factory.register(DubinSelect, BoundDubinSelect)
    factory.register(ButcherTableau, BoundButcherTableau)
    
    dynamics_config = {
        'mu': 1.05263157895,
        'sigma': 0.1,
        'num_samples': 500,
        'dt': 0.1,
        'velocity': 1.0,
        'horizon': 10,
        'initial_set': 'left',
        'unsafe_set': 'walls'
    }
    dynamics = DubinsCarStrategyComposition(dynamics_config, DubinsCarNoActuation())
    bound = factory.build(dynamics.nominal_system)

    x_space = torch.linspace(-2.0, 2.0, 41)
    x_cell_width = (x_space[1] - x_space[0]) / 2
    x_slice_centers = (x_space[:-1] + x_space[1:]) / 2

    phi_space = torch.linspace(-np.pi / 2, np.pi / 2, 41)
    phi_cell_width = (phi_space[1] - phi_space[0]) / 2
    phi_slice_centers = (phi_space[:-1] + phi_space[1:]) / 2

    cell_width = torch.stack([x_cell_width, x_cell_width, phi_cell_width], dim=-1)
    cell_centers = torch.cartesian_prod(x_slice_centers, x_slice_centers, phi_slice_centers)

    crown_bounds = bound.crown(HyperRectangle.from_eps(cell_centers, cell_width))

    print('Done computing bounds')
    obj = {
        'num_partitions': crown_bounds.region.lower.size(0),
        'lowerA': crown_bounds.lower[0].tolist(),
        'lower_bias': crown_bounds.lower[1].tolist(),
        'upperA': crown_bounds.upper[0].tolist(),
        'upper_bias': crown_bounds.upper[1].tolist(),
        'region_lower': crown_bounds.region.lower.tolist(),
        'region_upper': crown_bounds.region.upper.tolist(),
    }
    json_obj = json.dumps(obj, indent=4)
    
    with open('../../julia/AutonomousVehicles/models/dubin_bounds.json', 'w') as outfile:
        outfile.write(json_obj)


if __name__ == '__main__':
    export_bounds()
    # plot_dubins_car()
    # plot_dubins_car_fixed_strategy_div()
    # plot_dubins_car_fixed_strategy_x_sin_phi()
    # plot_dubins_car_fixed_strategy_y_cos_phi()

    # Given that these four shows true bounds and the rest is simple interval arithmetic, I believe IBP for both the
    # dynamics and the fixed strategy.
