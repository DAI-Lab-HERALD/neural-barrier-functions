import math
from typing import Tuple

import numpy as np
import torch
from bound_propagation import BoundModule, IntervalBounds, LinearBounds, Cat
from bound_propagation.activation import assert_bound_order, regimes
from torch import nn
from torch.distributions import Normal

from learned_cbf.dynamics import StochasticDynamics
from learned_cbf.utils import overlap_circle, overlap_rectangle, overlap_outside_circle


class DubinsCar(StochasticDynamics, nn.Module):
    def __init__(self, dynamics_config):
        StochasticDynamics.__init__(self, dynamics_config['num_samples'])
        nn.Module.__init__(self)

        self.dt = dynamics_config['dt']
        self.velocity = dynamics_config['velocity']

        dist = Normal(0.0, torch.tensor(dynamics_config['sigma']))
        self.register_buffer('z', dist.sample((self.num_samples, 3)).view(-1, 1, 3))

    def forward(self, x):
        x1 = x[..., 0] + self.dt * (self.velocity * x[..., 2].sin() + self.z[..., 0])
        x2 = x[..., 1] + self.dt * (self.velocity * x[..., 2].cos() + self.z[..., 1])
        # x[..., 3] = u, i.e. the control. We assume it's concatenated on the last dimension
        x3 = x[..., 2] + self.dt * (x[..., 3] + self.z[..., 2])

        x = torch.stack([x1, x2, x3], dim=-1)
        return x

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


class BoundDubinsCar(BoundModule):
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

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        x1_lower = bounds.lower[..., 0] + self.module.dt * (self.module.velocity * bounds.lower[..., 2].sin() + self.module.z[..., 0])
        x1_upper = bounds.upper[..., 0] + self.module.dt * (self.module.velocity * bounds.upper[..., 2].sin() + self.module.z[..., 0])

        x2_lower = bounds.lower[..., 1] + self.module.dt * self.module.velocity * (torch.min(bounds.lower[..., 2].cos(), bounds.upper[..., 2].cos()) + self.module.z[..., 1])

        across_center = (bounds.lower[..., 2] <= 0.0) & (bounds.upper[..., 2] >= 0.0)
        center_max = across_center * torch.ones_like(bounds.upper[..., 1])
        boundary_max = torch.max(bounds.lower[..., 2].cos(), bounds.upper[..., 2].cos())
        x2_upper = bounds.upper[..., 1] + self.module.dt * (self.module.velocity * torch.max(boundary_max, center_max) + self.module.z[..., 1])

        # x[..., 3] = u, i.e. the control. We assume it's concatenated on the last dimension
        x3_lower = bounds.lower[..., 2] + self.module.dt * (bounds.lower[..., 3] + self.module.z[..., 2])
        x3_upper = bounds.upper[..., 2] + self.module.dt * (bounds.upper[..., 3] + self.module.z[..., 2])

        lower = torch.stack([x1_lower, x2_lower, x3_lower], dim=-1)
        upper = torch.stack([x1_upper, x2_upper, x3_upper], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        assert in_size == 4

        return 3


class DubinsFixedStrategy(nn.Module):
    def forward(self, x):
        u = -x[..., 2].sin() + 3 * (x[..., 0] * x[..., 2].sin() + x[..., 1] * x[..., 2].cos()) / (0.5 + x[..., 0]**2 + x[..., 1]**2)
        return u.unsqueeze(-1)


class BoundDubinsFixedStrategy(BoundModule):
    @property
    def need_relaxation(self):
        return False

    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

    def crown_backward(self, linear_bounds):
        raise NotImplementedError()

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        lower = -bounds.upper[..., 2].sin()
        upper = -bounds.lower[..., 2].sin()

        across_center1 = (bounds.lower[..., 0] <= 0) & (bounds.upper[..., 0] >= 0)
        across_center2 = (bounds.lower[..., 1] <= 0) & (bounds.upper[..., 1] >= 0)

        min1 = torch.min(bounds.lower[..., 0] ** 2, bounds.upper[..., 0] ** 2) * (~across_center1)
        min2 = torch.min(bounds.lower[..., 1] ** 2, bounds.upper[..., 1] ** 2) * (~across_center2)

        max1 = torch.max(bounds.lower[..., 0] ** 2, bounds.upper[..., 0] ** 2)
        max2 = torch.max(bounds.lower[..., 1] ** 2, bounds.upper[..., 1] ** 2)

        lower_div = 1.0 / (0.5 + max1 + max2)
        upper_div = 1.0 / (0.5 + min1 + min2)

        x_sin_phi = torch.stack([
            bounds.lower[..., 0] * bounds.lower[..., 2].sin(),
            bounds.lower[..., 0] * bounds.upper[..., 2].sin(),
            bounds.upper[..., 0] * bounds.lower[..., 2].sin(),
            bounds.upper[..., 0] * bounds.upper[..., 2].sin(),
        ], dim=-1)
        lower_x_sin_phi = torch.min(x_sin_phi, dim=-1).values
        upper_x_sin_phi = torch.max(x_sin_phi, dim=-1).values

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


class DubinsCarFixedStrategyComposition(nn.Sequential, StochasticDynamics):
    def __init__(self, dynamics_config):
        StochasticDynamics.__init__(self, dynamics_config['num_samples'])
        nn.Sequential.__init__(self,
            Cat(DubinsFixedStrategy()),
            DubinsCar(dynamics_config)
        )

    def initial(self, x, eps=None):
        return self[1].initial(x, eps)

    def safe(self, x, eps=None):
        return self[1].safe(x, eps)

    def unsafe(self, x, eps=None):
        return self[1].unsafe(x, eps)

    def state_space(self, x, eps=None):
        return self[1].state_space(x, eps)

    @property
    def volume(self):
        return self[1].volume