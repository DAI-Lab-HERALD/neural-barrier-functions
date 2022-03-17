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

    def safe(self, x):
        if self.safe_set_type == 'circle':
            return x.norm(dim=-1) <= 2.0
        elif self.safe_set_type == 'annulus':
            return (x.norm(dim=-1) <= 4.0) & (x.norm(dim=-1) >= 0.5)
        else:
            raise ValueError('Invalid safe set for population')

    def initial(self, x):
        if self.safe_set_type == 'circle':
            return x.norm(dim=-1) <= 1.0
        elif self.safe_set_type == 'annulus':
            return (x.norm(dim=-1) >= 2.0) & (x.norm(dim=-1) <= 2.5)
        else:
            raise ValueError('Invalid safe set for population')

    def state_space(self, x):
        if self.safe_set_type == 'circle':
            return (x[..., 0].abs() <= 3.0) & (x[..., 1].abs() <= 3.0)
        elif self.safe_set_type == 'annulus':
            return (x[..., 0].abs() <= 4.5) & (x[..., 1].abs() <= 4.5)
        else:
            raise ValueError('Invalid safe set for population')

