import math
from typing import Tuple

import torch
from bound_propagation import BoundModule, IntervalBounds, LinearBounds
from bound_propagation.activation import assert_bound_order, regimes
from torch import nn
from torch.distributions import Normal

from learned_cbf.dynamics import StochasticDynamics


def overlap_rectangle(partition_lower, partition_upper, rect_lower, rect_upper):
    # Separating axis theorem
    return (partition_upper[..., 0] >= rect_lower[0]) & (partition_lower[..., 0] <= rect_upper[0]) & \
           (partition_upper[..., 1] >= rect_lower[1]) & (partition_lower[..., 1] <= rect_upper[1])


def overlap_circle(partition_lower, partition_upper, center, radius):
    closest_point = torch.max(partition_lower, torch.min(partition_upper, center))
    distance = (closest_point - center).norm(dim=-1)
    return distance <= radius


def overlap_outside_rectangle(partition_lower, partition_upper, rect_lower, rect_upper):
    return (partition_upper[..., 0] >= rect_upper[0]) | (partition_lower[..., 0] <= rect_lower[0]) | \
           (partition_upper[..., 1] >= rect_upper[1]) | (partition_lower[..., 1] <= rect_lower[1])


def overlap_outside_circle(partition_lower, partition_upper, center, radius):
    farthest_point = torch.where((partition_lower - center).abs() > (partition_upper - center).abs(), partition_lower, partition_upper)
    distance = (farthest_point - center).norm(dim=-1)
    return distance >= radius


class Polynomial(StochasticDynamics, nn.Module):
    def __init__(self, dynamics_config):
        StochasticDynamics.__init__(self, dynamics_config['num_samples'])
        nn.Module.__init__(self)

        self.dt = dynamics_config['dt']

        dist = Normal(0.0, dynamics_config['sigma'])
        self.register_buffer('z', dist.sample((self.num_samples,)).view(-1, 1))

    def forward(self, x):
        x1 = self.dt * x[..., 1] + self.z
        x2 = self.dt * ((x[..., 0] ** 3) / 3.0 - x[..., 0] - x[..., 1]).unsqueeze(0).expand_as(x1)

        x = torch.stack([x1, x2], dim=-1)
        return x

    def initial(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_circle(lower_x, upper_x, torch.tensor([1.5, 0]), math.sqrt(0.25))
        cond2 = overlap_rectangle(lower_x, upper_x, torch.tensor([-1.8, -0.1]), torch.tensor([-1.2, 0.1]))
        cond3 = overlap_rectangle(lower_x, upper_x, torch.tensor([-1.4, -0.5]), torch.tensor([-1.2, 0.1]))

        return cond1 | cond2 | cond3

    def safe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_outside_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0]), math.sqrt(0.16))
        cond2 = overlap_outside_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.6, 0.5]))
        cond3 = overlap_outside_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.8, 0.3]))

        return cond1 & cond2 & cond3

    def unsafe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0]), math.sqrt(0.16))
        cond2 = overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.6, 0.5]))
        cond3 = overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.8, 0.3]))

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


# @torch.jit.script
def _delta(W_tilde: torch.Tensor, beta_lower: torch.Tensor, beta_upper: torch.Tensor) -> torch.Tensor:
    return torch.where(W_tilde < 0, beta_lower.unsqueeze(-2), beta_upper.unsqueeze(-2))


# @torch.jit.script
def _lambda(W_tilde: torch.Tensor, alpha_lower: torch.Tensor, alpha_upper: torch.Tensor) -> torch.Tensor:
    return torch.where(W_tilde < 0, alpha_lower.unsqueeze(-2), alpha_upper.unsqueeze(-2))


# @torch.jit.script
def crown_backward_polynomial_jit(W_tilde: torch.Tensor, alpha: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    bias = torch.sum(W_tilde * _delta(W_tilde, *beta), dim=-1)
    W_tilde = W_tilde * _lambda(W_tilde, *alpha)

    return W_tilde, bias


class BoundPolynomial(BoundModule):
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
            lower = crown_backward_polynomial_jit(linear_bounds.lower[0][..., 0], alpha, beta)

            lowerA = torch.cat([torch.ones_like(lower[0]), lower[0]], dim=-1)
            lower_bias = linear_bounds.lower[1] + torch.cat([torch.zeros_like(lower[1]), lower[1]], dim=-1)
            lower = (lowerA, lower_bias)

        if linear_bounds.upper is None:
            upper = None
        else:
            alpha = self.alpha_lower, self.alpha_upper
            beta = self.beta_lower, self.beta_upper
            upper = crown_backward_polynomial_jit(linear_bounds.upper[0][..., 0], alpha, beta)

            upperA = torch.cat([torch.ones_like(upper[0]), upper[0]], dim=-1)
            upper_bias = linear_bounds.upper[1] + torch.cat([torch.zeros_like(upper[1]), upper[1]], dim=-1)
            upper = (upperA, upper_bias)

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        x1_lower = self.module.dt * bounds.lower[..., 1] + self.module.z
        x1_upper = self.module.dt * bounds.upper[..., 1] + self.module.z

        # x[..., 0] ** 3 is non-decreasing (and multiplying/dividing by a positive constant preserves this)
        x1_cubed_lower, x1_cubed_upper = (bounds.lower[..., 0] ** 3) / 3.0, (bounds.upper[..., 0] ** 3) / 3.0

        x2_lower = self.module.dt * (x1_cubed_lower - bounds.upper.sum(dim=-1)).unsqueeze(0).expand_as(x1_lower)
        x2_upper = self.module.dt * (x1_cubed_upper - bounds.lower.sum(dim=-1)).unsqueeze(0).expand_as(x1_lower)

        lower = torch.stack([x1_lower, x2_lower], dim=-1)
        upper = torch.stack([x1_upper, x2_upper], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        assert in_size == 2

        return 2
