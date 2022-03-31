import math
from typing import Tuple

import torch
from bound_propagation import BoundModule, IntervalBounds, LinearBounds, HyperRectangle, BoundModelFactory
from bound_propagation.activation import assert_bound_order, regimes
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Normal

from euler import Euler, BoundEuler
from learned_cbf.dynamics import StochasticDynamics
from learned_cbf.utils import overlap_circle, overlap_rectangle, overlap_outside_circle, overlap_outside_rectangle


class PolynomialUpdate(nn.Module):
    def __init__(self, dynamics_config):
        super().__init__()

        dist = Normal(0.0, dynamics_config['sigma'])
        self.register_buffer('z', dist.sample((dynamics_config['num_samples'],)).view(-1, 1))

    def x1_cubed(self, x):
        return (x[..., 0] ** 3) / 3.0

    def forward(self, x):
        x1 = x[..., 1] + self.z
        x2 = (self.x1_cubed(x) - x.sum(dim=-1)).unsqueeze(0).expand_as(x1)

        x = torch.stack([x1, x2], dim=-1)
        return x


@torch.jit.script
def crown_backward_polynomial_jit(W_tilde: torch.Tensor, z: torch.Tensor, alpha: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    _lambda = torch.where(W_tilde[..., 1] < 0, alpha[0].unsqueeze(-1), alpha[1].unsqueeze(-1))
    _delta = torch.where(W_tilde[..., 1] < 0, beta[0].unsqueeze(-1), beta[1].unsqueeze(-1))

    bias = W_tilde[..., 1] * _delta + z.unsqueeze(-1) * W_tilde[..., 0]
    W_tilde = torch.stack([W_tilde[..., 1] * _lambda - W_tilde[..., 1], W_tilde[..., 0] - W_tilde[..., 1]], dim=-1)

    return W_tilde, bias


class BoundPolynomialUpdate(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None
        self.bounded = False

    def func(self, x):
        return x**3 / 3.0

    def derivative(self, x):
        return x**2

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

        def add_linear(alpha, beta, mask, a, x, y):
            alpha[mask] = a[mask]
            beta[mask] = y[mask] - a[mask] * x[mask]

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
        direct_upper = np & (slope >= lower_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct_upper, a=slope, x=upper, y=upper_act)

        # Else use polynomial derivative to find upper bound on slope.
        implicit_upper = np & (slope < lower_prime)

        d = -upper / 2.0
        # Slope has to attach to (upper, upper^3)
        add_linear(self.alpha_upper, self.beta_upper, mask=implicit_upper, a=self.derivative(d), x=upper, y=upper_act)

        # Lower bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct_lower = np & (slope >= upper_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct_lower, a=slope, x=lower, y=lower_act)

        # Else use polynomial derivative to find upper bound on slope.
        implicit_lower = np & (slope < upper_prime)

        d = -lower / 2.0
        # Slope has to attach to (lower, lower^3)
        add_linear(self.alpha_lower, self.beta_lower, mask=implicit_lower, a=self.derivative(d), x=lower, y=lower_act)

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

        # NOTE: The order of alpha and beta are deliberately reversed - this is not a mistake!
        if linear_bounds.lower is None:
            lower = None
        else:
            alpha = self.alpha_upper, self.alpha_lower
            beta = self.beta_upper, self.beta_lower
            lower = crown_backward_polynomial_jit(linear_bounds.lower[0], self.module.z, alpha, beta)

            lower = (lower[0], lower[1] + linear_bounds.upper[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            alpha = self.alpha_lower, self.alpha_upper
            beta = self.beta_lower, self.beta_upper
            upper = crown_backward_polynomial_jit(linear_bounds.upper[0], self.module.z, alpha, beta)
            upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward_x1_cubed(self, bounds):
        # x[..., 0] ** 3 is non-decreasing (and multiplying/dividing by a positive constant preserves this)
        return (bounds.lower[..., 0] ** 3) / 3.0, (bounds.upper[..., 0] ** 3) / 3.0

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        x1_lower = bounds.lower[..., 1] + self.module.z
        x1_upper = bounds.upper[..., 1] + self.module.z

        x1_cubed_lower, x1_cubed_upper = self.ibp_forward_x1_cubed(bounds)

        x2_lower = (x1_cubed_lower - bounds.upper.sum(dim=-1)).unsqueeze(0).expand_as(x1_lower)
        x2_upper = (x1_cubed_upper - bounds.lower.sum(dim=-1)).unsqueeze(0).expand_as(x1_lower)

        lower = torch.stack([x1_lower, x2_lower], dim=-1)
        upper = torch.stack([x1_upper, x2_upper], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        assert in_size == 2

        return 2


class Polynomial(Euler, StochasticDynamics):
    def __init__(self, dynamics_config):
        StochasticDynamics.__init__(self, dynamics_config['num_samples'])
        Euler.__init__(self, PolynomialUpdate(dynamics_config), dynamics_config['dt'])

    def initial(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_circle(lower_x, upper_x, torch.tensor([1.5, 0], device=x.device), math.sqrt(0.25))
        cond2 = overlap_rectangle(lower_x, upper_x, torch.tensor([-1.8, -0.1], device=x.device), torch.tensor([-1.2, 0.1], device=x.device))
        cond3 = overlap_rectangle(lower_x, upper_x, torch.tensor([-1.4, -0.5], device=x.device), torch.tensor([-1.2, 0.1], device=x.device))

        return cond1 | cond2 | cond3

    def sample_initial(self, num_particles):
        raise NotImplementedError()

    def safe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_outside_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0], device=x.device), math.sqrt(0.16))
        cond2 = overlap_outside_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1], device=x.device), torch.tensor([0.6, 0.5], device=x.device))
        cond3 = overlap_outside_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1], device=x.device), torch.tensor([0.8, 0.3], device=x.device))

        return cond1 & cond2 & cond3

    def unsafe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0], device=x.device), math.sqrt(0.16))
        cond2 = overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1], device=x.device), torch.tensor([0.6, 0.5], device=x.device))
        cond3 = overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1], device=x.device), torch.tensor([0.8, 0.3], device=x.device))

        return cond1 | cond2 | cond3

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return (upper_x[..., 0] >= -3.5) & (lower_x[..., 0] <= 2.0) &\
               (upper_x[..., 1] >= -2.0) & (lower_x[..., 1] <= 1.0)

    @property
    def volume(self):
        return (2.0 - (-3.5)) * (1.0 - (-2.0))


def plot_polynomial():
    dynamics = PolynomialUpdate({
        'sigma': 0.1,
        'num_samples': 500
    })

    bound = BoundPolynomialUpdate(dynamics, None)

    x1_space = torch.linspace(-3.5, 2.0, 5)
    x1_cell_width = (x1_space[1] - x1_space[0]) / 2
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2

    x2_space = torch.linspace(-2.0, 1.0, 5)
    x2_cell_width = (x2_space[1] - x2_space[0]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    cell_width = torch.stack([x1_cell_width, x2_cell_width], dim=-1)
    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)

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
        y_lower = crown_bounds.lower[0][i, 0, 0] * x1 + crown_bounds.lower[0][i, 0, 1] * x2 + crown_bounds.lower[1][0, i, 0]
        y_upper = crown_bounds.upper[0][i, 0, 0] * x1 + crown_bounds.upper[0][i, 0, 1] * x2 + crown_bounds.upper[1][0, i, 0]

        surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Plot function
        x1, x2 = input_bounds.lower[i], input_bounds.upper[i]
        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
        X = torch.cat(tuple(torch.dstack([x1, x2])))
        y = dynamics(X)[0, :, 0].view(50, 50)

        surf = ax.plot_surface(x1, x2, y, color='red', label='Polynomial - x', shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Bound propagation')
        plt.legend()

        plt.show()

        # Y
        x1, x2 = input_bounds.lower[i], input_bounds.upper[i]

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
        y_lower = crown_bounds.lower[0][i, 1, 0] * x1 + crown_bounds.lower[0][i, 1, 1] * x2 + crown_bounds.lower[1][0, i, 1]
        y_upper = crown_bounds.upper[0][i, 1, 0] * x1 + crown_bounds.upper[0][i, 1, 1] * x2 + crown_bounds.upper[1][0, i, 1]

        surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Plot function
        x1, x2 = input_bounds.lower[i], input_bounds.upper[i]
        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
        X = torch.cat(tuple(torch.dstack([x1, x2])))
        y = dynamics(X)[0, :, 1].view(50, 50)

        surf = ax.plot_surface(x1, x2, y, color='red', label='Polynomial - y', shade=False)
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
