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

    def corners(self, x, eps):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            bottom_left = torch.min(lower_x.abs(), upper_x.abs())
            bottom_right = torch.stack([torch.max(lower_x[..., 0].abs(), upper_x[..., 0].abs()), torch.min(lower_x[..., 1].abs(), upper_x[..., 1].abs())], dim=-1)
            top_left = torch.stack([torch.min(lower_x[..., 0].abs(), upper_x[..., 0].abs()), torch.max(lower_x[..., 1].abs(), upper_x[..., 1].abs())], dim=-1)
            top_right = torch.max(lower_x.abs(), upper_x.abs())
        else:
            bottom_left, bottom_right, top_left, top_right = x, x, x, x

        return bottom_left, bottom_right, top_left, top_right

    def initial(self, x, eps=None):
        bottom_left, bottom_right, top_left, top_right = self.corners(x, eps)

        if self.safe_set_type == 'circle':
            return bottom_left.norm(dim=-1) <= 1.0
        elif self.safe_set_type == 'annulus':
            return (top_right.sum(dim=-1) >= 4.0) & (bottom_left.sum(dim=-1) <= 5.0)
        else:
            raise ValueError('Invalid safe set for population')

    def sample_initial(self, num_particles):
        if self.safe_set_type == 'circle':
            dist = distributions.Uniform(0, 1)
            r = 1.0 * dist.sample((num_particles,)).sqrt()
            theta = dist.sample((num_particles,)) * 2 * np.pi

            return torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)
        elif self.safe_set_type == 'annulus':
            raise NotImplementedError()
        else:
            raise ValueError('Invalid safe set for population')

    def safe(self, x, eps=None):
        bottom_left, bottom_right, top_left, top_right = self.corners(x, eps)

        if self.safe_set_type == 'circle':
            return bottom_left.norm(dim=-1) <= 2.0
        elif self.safe_set_type == 'annulus':
            return (top_right.sum(dim=-1) >= 3.0) & (bottom_left.sum(dim=-1) <= 6.0)
        else:
            raise ValueError('Invalid safe set for population')

    def unsafe(self, x, eps=None):
        bottom_left, bottom_right, top_left, top_right = self.corners(x, eps)

        if self.safe_set_type == 'circle':
            return bottom_left.norm(dim=-1) >= 2.0
        elif self.safe_set_type == 'annulus':
            return (bottom_left.sum(dim=-1) <= 3.0) | (top_right.sum(dim=-1) >= 6.0)
        else:
            raise ValueError('Invalid safe set for population')

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        if self.safe_set_type == 'circle':
            return (upper_x[..., 0] >= -3.0) & (lower_x[..., 0] <= 3.0) & (upper_x[..., 1] >= -3.0) & (lower_x[..., 1] <= 3.0)
        elif self.safe_set_type == 'annulus':
            return (upper_x[..., 0] >= 0) & (lower_x[..., 0] <= 8.0) & (upper_x[..., 1] >= 0) & (lower_x[..., 1] <= 8.0)
        else:
            raise ValueError('Invalid safe set for population')

    @property
    def volume(self):
        if self.safe_set_type == 'circle':
            return 6.0 ** 2
        elif self.safe_set_type == 'annulus':
            return 4.5 ** 2
        else:
            raise ValueError('Invalid safe set for population')
