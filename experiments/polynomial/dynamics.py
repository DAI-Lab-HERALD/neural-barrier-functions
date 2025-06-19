import math
from typing import Tuple

import numpy as np
import torch
from bound_propagation import BoundModule, IntervalBounds, LinearBounds, HyperRectangle, BoundModelFactory
from bound_propagation.activation import assert_bound_order, regimes
from matplotlib import pyplot as plt
from torch import nn, distributions
from torch.distributions import Normal

from neural_barrier_functions.discretization import Euler, RK4
from neural_barrier_functions.dynamics import StochasticDynamics, AdditiveGaussianDynamics
from neural_barrier_functions.utils import overlap_circle, overlap_rectangle, overlap_outside_circle, overlap_outside_rectangle



class NominalPolynomialUpdate(nn.Module):
    def __init__(self, dynamics_config):
        super().__init__()

    def x1_cubed(self, x):
        return (x[..., 0] ** 3) / 3.0

    def forward(self, x):
        x1 = x[..., 1]
        x2 = self.x1_cubed(x) - x.sum(dim=-1)
        x = torch.stack([x1, x2], dim=-1)
        return x


@torch.jit.script
def crown_backward_nominal_polynomial_jit(W_lower: torch.Tensor, W_upper: torch.Tensor, alpha: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    _lambda = torch.where(W_upper[..., 1] < 0, alpha[0].unsqueeze(-1), alpha[1].unsqueeze(-1))
    _delta = torch.where(W_upper[..., 1] < 0, beta[0].unsqueeze(-1), beta[1].unsqueeze(-1))

    bias = W_upper[..., 1] * _delta

    W_tilde1 = W_upper[..., 1] * _lambda - W_lower[..., 1]
    W_tilde2 = W_upper[..., 0] - W_lower[..., 1]
    W_tilde = torch.stack([W_tilde1, W_tilde2], dim=-1)

    return W_tilde, bias


class BoundNominalPolynomialUpdate(BoundModule):
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
        zero_width, n, p, np = regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self.func(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self.func(upper[zero_width])

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

    def crown_backward(self, linear_bounds, optimize):
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

    def ibp_forward_x1_cubed(self, bounds):
        # x[..., 0] ** 3 is non-decreasing (and multiplying/dividing by a positive constant preserves this)
        return (bounds.lower[..., 0] ** 3) / 3.0, (bounds.upper[..., 0] ** 3) / 3.0
    
    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        x1_lower = bounds.lower[..., 1]
        x1_upper = bounds.upper[..., 1]

        x1_cubed_lower, x1_cubed_upper = self.ibp_forward_x1_cubed(bounds)

        x2_lower = x1_cubed_lower - bounds.upper.sum(dim=-1)
        x2_upper = x1_cubed_upper - bounds.lower.sum(dim=-1)

        lower = torch.stack([x1_lower, x2_lower], dim=-1)
        upper = torch.stack([x1_upper, x2_upper], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        assert in_size == 2

        return 2

class Polynomial(AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        self.dynamics_config = dynamics_config
        nominal = Euler(NominalPolynomialUpdate(dynamics_config), dynamics_config['dt'])
        print(dynamics_config)
        super().__init__(nominal, **dynamics_config)

    def initial(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_circle(lower_x, upper_x, torch.tensor([1.5, 0.0], device=x.device), math.sqrt(0.25))
        cond2 = overlap_rectangle(lower_x, upper_x, torch.tensor([-1.8, -0.1], device=x.device), torch.tensor([-1.2, 0.1], device=x.device))
        cond3 = overlap_rectangle(lower_x, upper_x, torch.tensor([-1.4, -0.5], device=x.device), torch.tensor([-1.2, 0.1], device=x.device))

        return cond1 | cond2 | cond3

    def sample_initial(self, num_particles):
        return self.sample_initial_unsafe(
            num_particles,
            (torch.tensor([-1.8, -0.1]), torch.tensor([-1.4, 0.1])),
            (torch.tensor([-1.4, -0.5]), torch.tensor([-1.2, 0.1])),
            (torch.tensor([1.5, 0.0]), math.sqrt(0.25))
        )

    def safe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_outside_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0], device=x.device), math.sqrt(0.16))
        cond2 = overlap_outside_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1], device=x.device), torch.tensor([0.6, 0.5], device=x.device))
        cond3 = overlap_outside_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1], device=x.device), torch.tensor([0.8, 0.3], device=x.device))

        return cond1 & cond2 & cond3

    def sample_safe(self, num_particles):
        samples = self.sample_state_space(2 * num_particles)
        samples = samples[self.safe(samples)]
        return samples[:num_particles]

    def unsafe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0], device=x.device), math.sqrt(0.16))
        cond2 = overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1], device=x.device), torch.tensor([0.6, 0.5], device=x.device))
        cond3 = overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1], device=x.device), torch.tensor([0.8, 0.3], device=x.device))

        return cond1 | cond2 | cond3

    def sample_unsafe(self, num_particles):
        return self.sample_initial_unsafe(
            num_particles,
            (torch.tensor([0.4, 0.3]), torch.tensor([0.6, 0.5])),
            (torch.tensor([0.4, 0.1]), torch.tensor([0.8, 0.3])),
            (torch.tensor([-1.0, -1.0]), math.sqrt(0.16))
        )

    def sample_initial_unsafe(self, num_particles, rect1, rect2, circle):
        rect1_area = (rect1[1] - rect1[0]).prod()
        rect2_area = (rect2[1] - rect2[0]).prod()
        circle_area = circle[1] ** 2 * np.pi
        total_area = rect1_area + rect2_area + circle_area

        rect1_prob = rect1_area / total_area
        rect2_prob = rect2_area / total_area
        circle_prob = circle_area / total_area

        dist = distributions.Multinomial(total_count=num_particles, probs=torch.tensor([rect1_prob, rect2_prob, circle_prob]))
        count = dist.sample().int()

        dist = distributions.Uniform(rect1[0], rect1[1])
        rect1_samples = dist.sample((count[0],))

        dist = distributions.Uniform(rect2[0], rect2[1])
        rect2_samples = dist.sample((count[1],))

        dist = distributions.Uniform(0, 1)
        r = circle[1] * dist.sample((count[2],)).sqrt()
        theta = dist.sample((count[2],)) * 2 * np.pi
        circle_samples = circle[0] + torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)

        return torch.cat([rect1_samples, rect2_samples, circle_samples], dim=0)

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return (upper_x[..., 0] >= -3.5) & (lower_x[..., 0] <= 2.0) & \
               (upper_x[..., 1] >= -2.0) & (lower_x[..., 1] <= 1.0)

    def sample_state_space(self, num_particles):
        dist = distributions.Uniform(torch.tensor([-3.5, -2]), torch.tensor([2.0, 1.0]))
        return dist.sample((num_particles,))

    @property
    def volume(self):
        return (2.0 - (-3.5)) * (1.0 - (-2.0))
