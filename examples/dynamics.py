import torch
from torch import nn
from torch.distributions import Normal

from learned_cbf.dynamics import StochasticDynamics


class PopulationStep(nn.Linear):
    # x[1] = juveniles
    # x[2] = adults

    sigma = 0.1
    fertility_rate = 0.2
    survival_juvenile = 0.3
    survival_adult = 0.8

    def __init__(self, num_samples):
        super().__init__(2, 2)

        del self.weight
        del self.bias

        self.register_buffer('weight', torch.as_tensor([
            [0.0, self.fertility_rate],
            [self.survival_juvenile, self.survival_adult]
        ]), persistent=True)

        dist = Normal(0.0, self.sigma)
        z = dist.sample((num_samples,))
        self.register_buffer('bias', torch.stack([torch.zeros_like(z), z], dim=-1).unsqueeze(-2), persistent=True)


class Population(StochasticDynamics):
    def __init__(self, num_samples):
        super().__init__(
            PopulationStep(num_samples),
            num_samples=num_samples
        )
