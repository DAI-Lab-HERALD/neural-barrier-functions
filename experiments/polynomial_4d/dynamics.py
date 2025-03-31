import math
from typing import Tuple

import numpy as np
import torch
from bound_propagation import BoundModule, IntervalBounds, LinearBounds, HyperRectangle, BoundModelFactory
from bound_propagation.activation import assert_bound_order, regimes
from matplotlib import pyplot as plt
from torch import nn, distributions
from torch.distributions import Normal

from neural_barrier_functions.discretization import Euler
from neural_barrier_functions.dynamics import AdditiveGaussianDynamics
from neural_barrier_functions.utils import overlap_circle, overlap_rectangle, overlap_outside_circle, overlap_outside_rectangle


class PolynomialUpdate(nn.Module):
    def __init__(self, dynamics_config):
        super().__init__()

        dist = Normal(0.0, dynamics_config['sigma'])
        self.register_buffer('z', dist.sample((dynamics_config['num_samples'],)).view(-1, 1))

    def forward(self, x):
        x1 = x[..., 0] + x[..., 1] - x[..., 2] ** 3 + self.z
        x2 = x[..., 0] ** 2 + x[..., 1] - x[..., 2] - x[..., 3]
        x3 = -x[..., 0] + x[..., 1] ** 2 + x[..., 2]
        x4 = -x[..., 0] - x[..., 1]

        if x2.dim() != x1.dim():
            x2 = x2.unsqueeze(0).expand_as(x1)
            x3 = x3.unsqueeze(0).expand_as(x1)
            x4 = x4.unsqueeze(0).expand_as(x1)

        x = torch.stack([x1, x2, x3, x4], dim=-1)
        return x


@torch.jit.script
def crown_backward_polynomial_jit(W_lower: torch.Tensor, W_upper: torch.Tensor, z: torch.Tensor, alpha: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    _lambda_lower = torch.where(W_lower[..., :3] < 0, alpha[1].unsqueeze(-2), alpha[0].unsqueeze(-2))
    _delta_lower = torch.where(W_lower[..., :3] < 0, beta[1].unsqueeze(-2), beta[0].unsqueeze(-2))

    _lambda_upper = torch.where(W_upper[..., :3] < 0, alpha[0].unsqueeze(-2), alpha[1].unsqueeze(-2))
    _delta_upper = torch.where(W_upper[..., :3] < 0, beta[0].unsqueeze(-2), beta[1].unsqueeze(-2))

    bias = -W_lower[..., 0] * _delta_lower[..., 0] + W_upper[..., 1] * _delta_upper[..., 1] + W_upper[..., 2] * _delta_upper[ ..., 2] + z.unsqueeze(-1) * W_upper[..., 0]

    W_tilde1 = W_upper[..., 0] + W_upper[..., 1] * _lambda_upper[..., 1] - W_upper[..., 2] - W_lower[..., 3]
    W_tilde2 = W_upper[..., 0] + W_upper[..., 1] + W_upper[..., 2] * _lambda_upper[..., 2] - W_lower[..., 3]
    W_tilde3 = -W_lower[..., 0] * _lambda_lower[..., 0] - W_lower[..., 1] + W_upper[..., 2]
    W_tilde4 = -W_lower[..., 1]

    if W_tilde1.dim() != W_tilde2.dim():
        W_tilde2 = W_tilde2.unsqueeze(0).expand_as(W_tilde1)
        W_tilde3 = W_tilde3.unsqueeze(0).expand_as(W_tilde1)
        W_tilde4 = W_tilde4.unsqueeze(0).expand_as(W_tilde1)

    W_tilde = torch.stack([W_tilde1, W_tilde2, W_tilde3, W_tilde4], dim=-1)

    return W_tilde, bias


class BoundPolynomialUpdate(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None
        self.bounded = False

    def func(self, x):
        return torch.stack([x[..., 2] ** 3, x[..., 0] ** 2, x[..., 1] ** 2], dim=-1)

    def derivative(self, x):
        return torch.stack([3 * (x[..., 2] ** 2), 2 * x[..., 0], 2 * x[..., 1]], dim=-1)

    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower[..., [2, 0, 1]], preactivation.upper[..., [2, 0, 1]]
        zero_width, n, p, np = regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        lower_act, upper_act = self.func(lower), self.func(upper)
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, torch.min(lower_act[zero_width], upper_act[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, torch.max(lower_act[zero_width], upper_act[zero_width])

        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self.func(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, mask, order, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask, [order]]

            alpha[mask, [order]] = a
            beta[mask, [order]] = y[mask, [order]] - a * x[mask, [order]]

        ### x_1^2
        #######
        # All #
        #######
        all = (n | p | np)[..., 1]
        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=all, a=slope, x=lower, y=lower_act, order=1)

        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=all, a=d_prime, x=d, y=d_act, order=1)

        # ### x_2^2
        # #######
        # # All #
        # #######
        all = (n | p | np)[..., 2]
        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=all, a=slope, x=lower, y=lower_act, order=2)

        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=all, a=d_prime, x=d, y=d_act, order=2)

        ### x_3^3
        n, p, np = n[..., 0], p[..., 0], np[..., 0]
        ###################
        # Negative regime #
        ###################
        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=n, a=d_prime, x=d, y=d_act, order=0)

        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=n, a=slope, x=lower, y=lower_act, order=0)

        ###################
        # Positive regime #
        ###################
        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=p, a=d_prime, x=d, y=d_act, order=0)

        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=p, a=slope, x=upper, y=upper_act, order=0)

        #################
        # Crossing zero #
        #################
        # Upper bound #
        # If tangent to lower is below upper, then take direct slope between lower and upper
        direct_upper = np & (slope[..., 0] >= lower_prime[..., 0])
        add_linear(self.alpha_upper, self.beta_upper, mask=direct_upper, a=slope, x=upper, y=upper_act, order=0)

        # Else use polynomial derivative to find upper bound on slope.
        implicit_upper = np & (slope[..., 0] < lower_prime[..., 0])

        d = -upper / 2.0
        # Slope has to attach to (upper, upper^3)
        add_linear(self.alpha_upper, self.beta_upper, mask=implicit_upper, a=self.derivative(d), x=upper, y=upper_act, order=0)

        # Lower bound #
        # If tangent to upper is above lower, then take direct slope between lower and upper
        direct_lower = np & (slope[..., 0] >= upper_prime[..., 0])
        add_linear(self.alpha_lower, self.beta_lower, mask=direct_lower, a=slope, x=lower, y=lower_act, order=0)

        # Else use polynomial derivative to find upper bound on slope.
        implicit_lower = np & (slope[..., 0] < upper_prime[..., 0])

        d = -lower / 2.0
        # Slope has to attach to (lower, lower^3)
        add_linear(self.alpha_lower, self.beta_lower, mask=implicit_lower, a=self.derivative(d), x=lower, y=lower_act, order=0)

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

    def crown_backward(self, linear_bounds):
        assert self.bounded

        alpha = self.alpha_upper, self.alpha_lower
        beta = self.beta_upper, self.beta_lower
        lower = crown_backward_polynomial_jit(linear_bounds.upper[0], linear_bounds.lower[0], self.module.z, alpha, beta)

        lower = (lower[0], lower[1] + linear_bounds.lower[1])

        alpha = self.alpha_lower, self.alpha_upper
        beta = self.beta_lower, self.beta_upper
        upper = crown_backward_polynomial_jit(linear_bounds.lower[0], linear_bounds.upper[0], self.module.z, alpha, beta)
        upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward_squared(self, bounds, idx):
        lower, upper = bounds.lower[..., idx], bounds.upper[..., idx]
        lower_act, upper_act = lower ** 2, upper ** 2

        not_cross_zero = (upper <= 0) | (lower >= 0)

        lower_out = torch.min(lower_act, upper_act) * not_cross_zero
        upper_out = torch.max(lower_act, upper_act)

        return lower_out, upper_out

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        x1_lower = bounds.lower[..., 0] + bounds.lower[..., 1] - bounds.upper[..., 2] ** 3 + self.module.z
        x1_upper = bounds.upper[..., 0] + bounds.upper[..., 1] - bounds.lower[..., 2] ** 3 + self.module.z

        x1_squared_lower, x1_squared_upper = self.ibp_forward_squared(bounds, 0)
        x2_lower = x1_squared_lower + bounds.lower[..., 1] - bounds.upper[..., 2] - bounds.upper[..., 3]
        x2_upper = x1_squared_upper + bounds.upper[..., 1] - bounds.lower[..., 2] - bounds.lower[..., 3]

        x2_squared_lower, x2_squared_upper = self.ibp_forward_squared(bounds, 1)
        x3_lower = -bounds.upper[..., 0] + x2_squared_lower + bounds.lower[..., 2]
        x3_upper = -bounds.lower[..., 0] + x2_squared_upper + bounds.upper[..., 2]

        x4_lower = -bounds.upper[..., 0] - bounds.upper[..., 1]
        x4_upper = -bounds.lower[..., 0] - bounds.lower[..., 1]

        if x2_lower.dim() != x1_lower.dim():
            x2_lower = x2_lower.unsqueeze(0).expand_as(x1_lower)
            x2_upper = x2_upper.unsqueeze(0).expand_as(x1_upper)

            x3_lower = x3_lower.unsqueeze(0).expand_as(x1_lower)
            x3_upper = x3_upper.unsqueeze(0).expand_as(x1_upper)

            x4_lower = x4_lower.unsqueeze(0).expand_as(x1_lower)
            x4_upper = x4_upper.unsqueeze(0).expand_as(x1_upper)

        lower = torch.stack([x1_lower, x2_lower, x3_lower, x4_lower], dim=-1)
        upper = torch.stack([x1_upper, x2_upper, x3_upper, x4_upper], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        assert in_size == 4

        return 4


class NominalPolynomialUpdate(nn.Module):
    def __init__(self, dynamics_config):
        super().__init__()

        dist = Normal(0.0, dynamics_config['sigma'])

    def forward(self, x):
        x1 = x[..., 0] + x[..., 1] - x[..., 2] ** 3
        x2 = x[..., 0] ** 2 + x[..., 1] - x[..., 2] - x[..., 3]
        x3 = -x[..., 0] + x[..., 1] ** 2 + x[..., 2]
        x4 = -x[..., 0] - x[..., 1]

        x = torch.stack([x1, x2, x3, x4], dim=-1)
        return x


@torch.jit.script
def crown_backward_nominal_polynomial_jit(W_lower: torch.Tensor, W_upper: torch.Tensor, alpha: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    _lambda_lower = torch.where(W_lower[..., :2] < 0, alpha[1].unsqueeze(-2), alpha[0].unsqueeze(-2))
    _delta_lower = torch.where(W_lower[..., :2] < 0, beta[1].unsqueeze(-2), beta[0].unsqueeze(-2))

    _lambda_upper = torch.where(W_upper[..., :2] < 0, alpha[0].unsqueeze(-2), alpha[1].unsqueeze(-2))
    _delta_upper = torch.where(W_upper[..., :2] < 0, beta[0].unsqueeze(-2), beta[1].unsqueeze(-2))

    bias = -W_lower[..., 0] * _delta_lower[..., 2] + W_upper[..., 1] * _delta_upper[..., 0] + W_upper[..., 2] * _delta_upper[..., 1]

    W_tilde1 = W_upper[..., 0] + W_upper[..., 1] * _lambda_upper[..., 0] - W_upper[..., 2] - W_lower[..., 3]
    W_tilde2 = W_upper[..., 0] + W_upper[..., 1] + W_upper[..., 2] * _lambda_upper[..., 1] - W_lower[..., 3]
    W_tilde3 = -W_lower[..., 0] * _lambda_lower[..., 2] - W_lower[..., 1] + W_upper[..., 2]
    W_tilde4 = -W_lower[..., 1]

    W_tilde = torch.stack([W_tilde1, W_tilde2, W_tilde3, W_tilde4], dim=-1)

    return W_tilde, bias


class BoundNominalPolynomialUpdate(BoundPolynomialUpdate):
    def crown_backward(self, linear_bounds):
        assert self.bounded

        # NOTE: The order of alpha and beta are deliberately reversed - this is not a mistake!

        alpha = self.alpha_upper, self.alpha_lower
        beta = self.beta_upper, self.beta_lower
        lower = crown_backward_nominal_polynomial_jit(linear_bounds.upper[0], linear_bounds.lower[0], alpha, beta)
        lower = (lower[0], lower[1] + linear_bounds.lower[1])

        alpha = self.alpha_lower, self.alpha_upper
        beta = self.beta_lower, self.beta_upper
        upper = crown_backward_nominal_polynomial_jit(linear_bounds.lower[0], linear_bounds.upper[0], alpha, beta)
        upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        x1_lower = bounds.lower[..., 0] + bounds.lower[..., 1] - bounds.upper[..., 2] ** 3
        x1_upper = bounds.upper[..., 0] + bounds.upper[..., 1] - bounds.lower[..., 2] ** 3

        x1_squared_lower, x1_squared_upper = self.ibp_forward_squared(bounds, 0)
        x2_lower = x1_squared_lower + bounds.lower[..., 1] - bounds.upper[..., 2] - bounds.upper[..., 3]
        x2_upper = x1_squared_upper + bounds.upper[..., 1] - bounds.lower[..., 2] - bounds.lower[..., 3]

        x2_squared_lower, x2_squared_upper = self.ibp_forward_squared(bounds, 1)
        x3_lower = -bounds.upper[..., 0] + x2_squared_lower + bounds.lower[..., 2]
        x3_upper = -bounds.lower[..., 0] + x2_squared_upper + bounds.upper[..., 2]

        x4_lower = -bounds.upper[..., 0] - bounds.upper[..., 1]
        x4_upper = -bounds.lower[..., 0] - bounds.lower[..., 1]

        lower = torch.stack([x1_lower, x2_lower, x3_lower, x4_lower], dim=-1)
        upper = torch.stack([x1_upper, x2_upper, x3_upper, x4_upper], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)


class Polynomial(Euler, AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        self.dynamics_config = dynamics_config
        AdditiveGaussianDynamics.__init__(self, dynamics_config['num_samples'])
        Euler.__init__(self, PolynomialUpdate(dynamics_config), dynamics_config['dt'])

    @property
    def nominal_system(self):
        return Euler(NominalPolynomialUpdate(self.dynamics_config), self.dynamics_config['dt'])

    @property
    def v(self):
        return (
            torch.tensor([0.0, 0.0]),
            torch.tensor([self.dynamics_config['dt'] * self.dynamics_config['sigma'], 0.0])
        )

    def initial(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return overlap_circle(lower_x, upper_x, torch.tensor([0.0, 0.0, 0.0, 0.0], device=x.device), 0.2)

    def sample_initial(self, num_particles):
        dist = distributions.Uniform(torch.tensor([-0.2, -0.2, -0.2, -0.2]), torch.tensor([0.2, 0.2, 0.2, 0.2]))
        samples = dist.sample((8 * num_particles,))
        samples = samples[self.initial(samples)]

        return samples[:num_particles]

    def safe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return overlap_circle(lower_x, upper_x, torch.tensor([0.0, 0.0, 0.0, 0.0], device=x.device), 1.0)

    def sample_safe(self, num_particles):
        dist = distributions.Uniform(torch.tensor([-1.0, -1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0, 1.0]))
        samples = dist.sample((8 * num_particles,))
        samples = samples[self.safe(samples)]

        return samples[:num_particles]

    def unsafe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return overlap_outside_circle(lower_x, upper_x, torch.tensor([0.0, 0.0, 0.0, 0.0], device=x.device), 1.0)

    def sample_unsafe(self, num_particles):
        samples = self.sample_state_space(2 * num_particles)
        samples = samples[self.unsafe(samples)]
        return samples[:num_particles]

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return (upper_x[..., 0] >= -1.5) & (lower_x[..., 0] <= 1.5) & \
               (upper_x[..., 1] >= -1.5) & (lower_x[..., 1] <= 1.5) & \
               (upper_x[..., 2] >= -1.5) & (lower_x[..., 2] <= 1.5) & \
               (upper_x[..., 3] >= -1.5) & (lower_x[..., 3] <= 1.5)

    def sample_state_space(self, num_particles):
        dist = distributions.Uniform(torch.tensor([-1.5, -1.5, -1.5, -1.5]), torch.tensor([1.5, 1.5, 1.5, 1.5]))
        return dist.sample((num_particles,))

    @property
    def volume(self):
        return 3.0 ** 4


def plot_polynomial():
    dynamics = PolynomialUpdate({
        'sigma': 0.1,
        'num_samples': 500
    })

    bound = BoundPolynomialUpdate(dynamics, None)
    output_index = 1
    input_index1, input_index2 = 0, 1

    x_space = torch.linspace(-1.5, 1.5, 5)
    cell_width = (x_space[1] - x_space[0]) / 2
    slice_centers = (x_space[:-1] + x_space[1:]) / 2

    # cell_width = torch.stack([cell_width, cell_width, cell_width, cell_width], dim=-1)
    # cell_centers = torch.cartesian_prod(slice_centers, slice_centers, slice_centers, slice_centers)

    cell_width = torch.stack([cell_width, cell_width], dim=-1)
    cell_centers = torch.cartesian_prod(slice_centers, slice_centers)

    input_bounds = HyperRectangle.from_eps(cell_centers, cell_width)
    ibp_bounds = bound.ibp(input_bounds)
    crown_bounds = bound.crown_ibp(input_bounds)
    crown_interval = crown_bounds.concretize()

    for i in range(len(input_bounds)):
        # X
        x1, x2 = input_bounds.lower[i], input_bounds.upper[i]

        plt.clf()
        ax = plt.axes(projection='3d')

        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 10), torch.linspace(x1[1], x2[1], 10))

        # Plot IBP
        y1, y2 = ibp_bounds.lower[0, i, output_index].item(), ibp_bounds.upper[0, i, output_index].item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='yellow', label='IBP', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='yellow', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Plot LBP interval bounds
        y1, y2 = crown_interval.lower[0, i, output_index].item(), crown_interval.upper[0, i, output_index].item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Plot LBP linear bounds
        y_lower = crown_bounds.lower[0][i, output_index, input_index1] * x1 + crown_bounds.lower[0][i, output_index, input_index2] * x2 + crown_bounds.lower[1][0, i, output_index]
        y_upper = crown_bounds.upper[0][i, output_index, input_index1] * x1 + crown_bounds.upper[0][i, output_index, input_index2] * x2 + crown_bounds.upper[1][0, i, output_index]

        surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Plot function
        x1, x2 = input_bounds.lower[i], input_bounds.upper[i]

        x1, x2 = torch.meshgrid(torch.linspace(x1[input_index1], x2[input_index1], 50), torch.linspace(x1[input_index2], x2[input_index2], 50))
        x_zero = torch.zeros_like(x1)

        x1_prime = x1 if input_index1 == 0 else (x2 if input_index2 == 0 else x_zero)
        x2_prime = x1 if input_index1 == 1 else (x2 if input_index2 == 1 else x_zero)
        x3_prime = x1 if input_index1 == 2 else (x2 if input_index2 == 2 else x_zero)
        x4_prime = x1 if input_index1 == 3 else (x2 if input_index2 == 3 else x_zero)

        # X = torch.cat(tuple(torch.dstack([x1_prime, x2_prime, x3_prime, x4_prime])))
        X = torch.cat(tuple(torch.dstack([x1_prime, x2_prime])))
        y = dynamics(X)[0, :, output_index].view(50, 50)

        surf = ax.plot_surface(x1, x2, y, color='red', label='Polynomial - x', shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Bound propagation')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    plot_polynomial()
