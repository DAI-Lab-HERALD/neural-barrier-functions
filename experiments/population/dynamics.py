import torch
from torch import nn, Tensor
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

    def close_far(self, x, eps):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            closest_point = torch.min(lower_x.abs(), upper_x.abs())
            farthest_point = torch.max(lower_x.abs(), upper_x.abs())
        else:
            closest_point, farthest_point = x, x

        return closest_point, farthest_point

    def initial(self, x, eps=None):
        closest_point, farthest_point = self.close_far(x, eps)

        if self.safe_set_type == 'circle':
            return closest_point.norm(dim=-1) <= 1.0
        elif self.safe_set_type == 'annulus':
            return (farthest_point.norm(dim=-1) >= 2.0) & (closest_point.norm(dim=-1) <= 2.5)
        else:
            raise ValueError('Invalid safe set for population')

    def safe(self, x, eps=None):
        closest_point, farthest_point = self.close_far(x, eps)

        if self.safe_set_type == 'circle':
            return closest_point.norm(dim=-1) <= 2.0
        elif self.safe_set_type == 'annulus':
            return (farthest_point.norm(dim=-1) >= 1.0) & (closest_point.norm(dim=-1) <= 3.5)
        else:
            raise ValueError('Invalid safe set for population')

    def unsafe(self, x, eps=None):
        closest_point, farthest_point = self.close_far(x, eps)

        if self.safe_set_type == 'circle':
            return farthest_point.norm(dim=-1) >= 2.0
        elif self.safe_set_type == 'annulus':
            return (closest_point.norm(dim=-1) <= 1.0) | (farthest_point.norm(dim=-1) >= 3.5)
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
            return (upper_x[..., 0] >= 0) & (lower_x[..., 0] <= 4.5) & (upper_x[..., 1] >= 0) & (lower_x[..., 1] <= 4.5)
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
