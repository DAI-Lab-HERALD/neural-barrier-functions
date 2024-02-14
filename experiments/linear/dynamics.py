import numpy as np
import torch
from torch import nn, Tensor, distributions
from torch.distributions import Normal

from neural_barrier_functions.dynamics import AdditiveGaussianDynamics


class LinearDynamics(nn.Linear, AdditiveGaussianDynamics):
    @property
    def nominal_system(self):
        nominal = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            nominal.weight.copy_(self.weight.squeeze(0))

        return nominal

    @property
    def v(self):
        return torch.zeros_like(self.sigma), self.sigma

    def __init__(self, dynamics_config):
        nn.Linear.__init__(self, 2, 2)
        AdditiveGaussianDynamics.__init__(self, dynamics_config['num_samples'])

        del self.weight
        del self.bias

        self.register_buffer('weight', torch.as_tensor([
            [0.5, 0.0],
            [0.0, 0.5]
        ]).unsqueeze(0), persistent=True)

        self.sigma = torch.as_tensor(dynamics_config['sigma'])

        dist = Normal(torch.zeros((2,)), self.sigma)
        z = dist.sample((self.num_samples,))
        self.register_buffer('bias', z.unsqueeze(1), persistent=True)

        self.safe_set = dynamics_config['safe_set']

    def forward(self, input: Tensor) -> Tensor:
        dist = Normal(torch.zeros((2,)), self.sigma)
        z = dist.sample((self.num_samples,))
        self.bias = z.unsqueeze(1)

        return input.matmul(self.weight.transpose(-1, -2)) + self.bias

    def initial(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            outside1 = upper_x[..., 0] < -0.8
            outside2 = lower_x[..., 0] > -0.6
            outside3 = upper_x[..., 1] < -0.2
            outside4 = lower_x[..., 1] > 0.0

            return ~outside1 & ~outside2 & ~outside3 & ~outside4

        return (-0.8 <= x[..., 0]) & (x[..., 0] <= -0.6) & (-0.2 <= x[..., 1]) & (x[..., 1] <= 0.0)

    def sample_initial(self, num_particles):
        dist = torch.distributions.Uniform(torch.tensor([-0.8, -0.2]), torch.tensor([-0.6, 0.0]))
        x = dist.sample((num_particles,))

        return x

    def safe(self, x, eps=None):
        def in_obstacle1(p):
            # center and radius in L-inf norm sense
            center = torch.tensor([-0.55, 0.3], device=x.device)
            radius = 0.02
            return (p - center).abs().max(dim=-1).values < radius

        def in_obstacle2(p):
            # center and radius in L-inf norm sense
            center = torch.tensor([0.55, 0.15], device=x.device)
            radius = 0.02
            return (p - center).abs().max(dim=-1).values < radius

        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            outside1 = upper_x[..., 0] < -1.0
            outside2 = lower_x[..., 0] > 0.5
            outside3 = upper_x[..., 1] < -0.5
            outside4 = lower_x[..., 1] > 0.5

            inside_xs = ~outside1 & ~outside2 & ~outside3 & ~outside4

            if self.safe_set == 'convex':
                return inside_xs
            elif self.safe_set == 'non-convex':
                corner1 = lower_x
                corner2 = upper_x
                corner3 = torch.stack((lower_x[..., 0], upper_x[..., 1]), dim=-1)
                corner4 = torch.stack((upper_x[..., 0], lower_x[..., 1]), dim=-1)

                fully_inside_obstacle1 = in_obstacle1(corner1) & in_obstacle1(corner2) & in_obstacle1(corner3) & in_obstacle1(corner4)
                outside_obstacle1 = ~fully_inside_obstacle1

                fully_inside_obstacle2 = in_obstacle2(corner1) & in_obstacle2(corner2) & in_obstacle2(corner3) & in_obstacle2(corner4)
                outside_obstacle2 = ~fully_inside_obstacle2

                return inside_xs & outside_obstacle1 & outside_obstacle2
            else:
                raise ValueError(f'Invalid safe set type \'{self.safe_set}\'. Expected \'convex\' or \'non-convex\'')

        inside_xs = (-1.0 <= x[..., 0]) & (x[..., 0] <= 0.5) & (-0.5 <= x[..., 1]) & (x[..., 1] <= 0.5)
        if self.safe_set == 'convex':
            return inside_xs
        elif self.safe_set == 'non-convex':
            outside_obstacle1 = ~in_obstacle1(x)
            outside_obstacle2 = ~in_obstacle2(x)

            return inside_xs & outside_obstacle1 & outside_obstacle2
        else:
            raise ValueError(f'Invalid safe set type \'{self.safe_set}\'. Expected \'convex\' or \'non-convex\'')

    def sample_safe(self, num_particles):
        dist = torch.distributions.Uniform(torch.tensor([-1.0, -0.5]), torch.tensor([0.5, 0.5]))
        x = dist.sample((2 * num_particles,))

        x = x[self.safe(x)]

        return x[:num_particles]

    def unsafe(self, x, eps=None):
        # center and radius in L-inf norm sense
        center_obstacle1 = torch.tensor([-0.55, 0.3], device=x.device)
        radius_obstacle1 = 0.02

        center_obstacle2 = torch.tensor([0.55, 0.15], device=x.device)
        radius_obstacle2 = 0.02

        def in_obstacle1(p):
            return (p - center_obstacle1).abs().max(dim=-1).values < radius_obstacle1

        def in_obstacle2(p):
            return (p - center_obstacle2).abs().max(dim=-1).values < radius_obstacle2

        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            inside1 = lower_x[..., 0] <= -1.0
            inside2 = upper_x[..., 0] >= 0.5
            inside3 = lower_x[..., 1] <= -0.5
            inside4 = upper_x[..., 1] >= 0.5

            outside_xs = inside1 | inside2 | inside3 | inside4

            if self.safe_set == 'convex':
                return outside_xs
            elif self.safe_set == 'non-convex':
                x_near = torch.maximum(lower_x, torch.minimum(center_obstacle1, upper_x))
                inside_obstacle1 = in_obstacle1(x_near)

                radius_obstacle2 = 0.02

                x_near = torch.maximum(lower_x, torch.minimum(center_obstacle2, upper_x))
                inside_obstacle2 = in_obstacle2(x_near)

                return outside_xs | inside_obstacle1 | inside_obstacle2
            else:
                raise ValueError(f'Invalid safe set type \'{self.safe_set}\'. Expected \'convex\' or \'non-convex\'')

        outside_xs = (-1.0 >= x[..., 0]) | (x[..., 0] >= 0.5) | (-0.5 >= x[..., 1]) | (x[..., 1] >= 0.5)
        if self.safe_set == 'convex':
            return outside_xs
        elif self.safe_set == 'non-convex':
            in_obstacle1 = in_obstacle1(x)
            in_obstacle2 = in_obstacle2(x)

            return outside_xs | in_obstacle1 | in_obstacle2
        else:
            raise ValueError(f'Invalid safe set type \'{self.safe_set}\'. Expected \'convex\' or \'non-convex\'')

    def sample_unsafe(self, num_particles):
        x = self.sample_state_space(num_particles * 10)
        x = x[self.unsafe(x)]

        return x[:num_particles]

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            outside1 = upper_x[..., 0] < -1.5
            outside2 = lower_x[..., 0] > 1.0
            outside3 = upper_x[..., 1] < -1.0
            outside4 = lower_x[..., 1] > 1.0

            return ~outside1 & ~outside2 & ~outside3 & ~outside4

        return (-1.5 <= x[..., 0]) & (x[..., 0] <= 1.0) & (-1.0 <= x[..., 1]) & (x[..., 1] <= 1.0)

    def sample_state_space(self, num_particles):
        dist = torch.distributions.Uniform(torch.tensor([-1.5, -1.0]), torch.tensor([1.0, 1.0]))
        x = dist.sample((num_particles,))

        return x

    @property
    def volume(self):
        return 2.5 * 2.0
