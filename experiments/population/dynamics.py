import torch
from torch import nn
from torch.distributions import Normal

from learned_cbf.dynamics import StochasticDynamics


class PopulationStep(nn.Linear):
    # x[1] = juveniles
    # x[2] = adults

    def __init__(self, dynamics_config):
        super().__init__(2, 2)

        del self.weight
        del self.bias

        self.register_buffer('weight', torch.as_tensor([
            [0.0, dynamics_config['fertility_rate']],
            [dynamics_config['survival_juvenile'], dynamics_config['survival_adult']]
        ]), persistent=True)

        dist = Normal(0.0, dynamics_config['sigma'])
        z = dist.sample((dynamics_config['num_samples'],))
        self.register_buffer('bias', torch.stack([torch.zeros_like(z), z], dim=-1).unsqueeze(-2), persistent=True)


class Population(StochasticDynamics):
    def __init__(self, dynamics_config):
        super().__init__(
            PopulationStep(dynamics_config),
            num_samples=dynamics_config['num_samples']
        )

        self.safe_set_type = dynamics_config['safe_set']

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
            return (farthest_point.norm(dim=-1) >= 0.5) & (closest_point.norm(dim=-1) <= 4.0)
        else:
            raise ValueError('Invalid safe set for population')

    def unsafe(self, x, eps=None):
        closest_point, farthest_point = self.close_far(x, eps)

        if self.safe_set_type == 'circle':
            return farthest_point.norm(dim=-1) >= 2.0
        elif self.safe_set_type == 'annulus':
            return (closest_point.norm(dim=-1) <= 0.5) | (farthest_point.norm(dim=-1) >= 4.0)
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
            return (upper_x[..., 0] >= -4.5) & (lower_x[..., 0] <= 4.5) & (upper_x[..., 1] >= -4.5) & (lower_x[..., 1] <= 4.5)
        else:
            raise ValueError('Invalid safe set for population')

