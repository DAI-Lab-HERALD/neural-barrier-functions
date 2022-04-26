import numpy as np
import torch
from torch import nn, Tensor, distributions
from torch.distributions import Normal

from learned_cbf.dynamics import StochasticDynamics


class Population(nn.Linear, StochasticDynamics):
    # x[1] = juveniles
    # x[2] = adults

    def __init__(self, dynamics_config):
        nn.Linear.__init__(self, 2, 2)
        StochasticDynamics.__init__(self, dynamics_config['num_samples'])

        del self.weight
        del self.bias

        self.register_buffer('weight', torch.as_tensor([
            [0.0, dynamics_config['fertility_rate']],
            [dynamics_config['survival_juvenile'], dynamics_config['survival_adult']]
        ]).unsqueeze(0), persistent=True)

        dist = Normal(0.0, dynamics_config['sigma'])
        z = dist.sample((self.num_samples,))
        self.register_buffer('bias', torch.stack([torch.zeros_like(z), z], dim=-1).unsqueeze(1), persistent=True)

        self.safe_set_type = dynamics_config['safe_set']

    def forward(self, input: Tensor) -> Tensor:
        return input.matmul(self.weight.transpose(-1, 1)) + self.bias

    def near_far(self, x, eps):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            near = torch.min(lower_x.abs(), upper_x.abs())
            far = torch.max(lower_x.abs(), upper_x.abs())
        else:
            near, far = x, x

        return near, far

    def initial(self, x, eps=None):
        near, far = self.near_far(x, eps)

        if self.safe_set_type == 'circle':
            return near.norm(dim=-1) <= 1.5
        elif self.safe_set_type == 'stripe':
            return (far.sum(dim=-1) >= 4.24999) & (near.sum(dim=-1) <= 4.75001) & (far[..., 0] >= 2.24999) & (far[..., 1] >= 2.24999) & (near[..., 0] <= 2.50001) & (near[..., 1] <= 2.50001)
        else:
            raise ValueError('Invalid safe set for population')

    def sample_initial(self, num_particles):
        if self.safe_set_type == 'circle':
            dist = distributions.Uniform(0, 1)
            r = 1.5 * dist.sample((num_particles,)).sqrt()
            theta = dist.sample((num_particles,)) * 2 * np.pi

            return torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)
        elif self.safe_set_type == 'stripe':
            dist = torch.distributions.Uniform(torch.tensor([2.0, 2.0]), torch.tensor([2.75, 2.75]))
            x = dist.sample((num_particles * 4,))
            x = x[self.initial(x)]

            return x[:num_particles]
        else:
            raise ValueError('Invalid safe set for population')

    def safe(self, x, eps=None):
        near, far = self.near_far(x, eps)

        if self.safe_set_type == 'circle':
            return near.norm(dim=-1) <= 2.0
        elif self.safe_set_type == 'stripe':
            return (far.sum(dim=-1) >= 0.99999) & (near.sum(dim=-1) <= 8.00001) & (far[..., 0] >= 0.49999) & (far[..., 1] >= 0.49999) & (near[..., 0] <= 7.50001) & (near[..., 1] <= 7.50001)
        else:
            raise ValueError('Invalid safe set for population')

    def sample_safe(self, num_particles):
        if self.safe_set_type == 'circle':
            dist = distributions.Uniform(0, 1)
            r = 2.0 * dist.sample((num_particles,)).sqrt()
            theta = dist.sample((num_particles,)) * 2 * np.pi

            return torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)
        elif self.safe_set_type == 'stripe':
            x = self.sample_state_space(num_particles * 4)
            dist = torch.distributions.Uniform(torch.tensor([1.5, 1.5]), torch.tensor([4.5, 4.5]))
            x = dist.sample((num_particles * 4,))
            x = x[self.safe(x)]

            return x[:num_particles]
        else:
            raise ValueError('Invalid safe set for population')

    def unsafe(self, x, eps=None):
        near, far = self.near_far(x, eps)

        if self.safe_set_type == 'circle':
            return far.norm(dim=-1) >= 2.0
        elif self.safe_set_type == 'stripe':
            return (near.sum(dim=-1) <= 1.00001) | (far.sum(dim=-1) >= 7.99999) | (near[..., 0] <= 0.50001) | (near[..., 1] <= 0.50001) | (far[..., 0] >= 7.49999) | (far[..., 1] >= 7.49999)
        else:
            raise ValueError('Invalid safe set for population')

    def sample_unsafe(self, num_particles):
        x = self.sample_state_space(num_particles * 4)
        x = x[self.unsafe(x)]

        return x[:num_particles]

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        if self.safe_set_type == 'circle':
            return (upper_x[..., 0] >= -3.0) & (lower_x[..., 0] <= 3.0) & (upper_x[..., 1] >= -3.0) & (lower_x[..., 1] <= 3.0)
        elif self.safe_set_type == 'stripe':
            return (upper_x[..., 0] >= 0) & (lower_x[..., 0] <= 8.0) & (upper_x[..., 1] >= 0) & (lower_x[..., 1] <= 8.0)
        else:
            raise ValueError('Invalid safe set for population')

    def sample_state_space(self, num_particles):
        if self.safe_set_type == 'circle':
            dist = torch.distributions.Uniform(torch.tensor([-3.0, -3.0]), torch.tensor([3.0, 3.0]))
        elif self.safe_set_type == 'stripe':
            dist = torch.distributions.Uniform(torch.tensor([0.0, 0.0]), torch.tensor([8.0, 8.0]))
        else:
            raise ValueError('Invalid safe set for population')

        return dist.sample((num_particles,))

    @property
    def volume(self):
        if self.safe_set_type == 'circle':
            return 6.0 ** 2
        elif self.safe_set_type == 'stripe':
            return 8.0 ** 2
        else:
            raise ValueError('Invalid safe set for population')
