from functools import partial

import torch
from torch.distributions import Normal

from learned_cbf.dynamics import StochasticDynamics


class Population(StochasticDynamics):
    # x[1] = juveniles
    # x[2] = adults

    sigma = 0.1
    fertility_rate = 0.2
    survival_juvenile = 0.3
    survival_adult = 0.8

    def dynamics(self, x, z):
        A = torch.as_tensor([
            [0.0, self.fertility_rate],
            [self.survival_juvenile, self.survival_adult]
        ], device=x.device)

        return A.matmul(x.unsqueeze(-1))[..., 0] + torch.stack([torch.zeros_like(z), z], dim=-1)

    def sample(self, num_samples):
        dist = Normal(0.0, self.sigma)
        z = dist.sample(num_samples)

        return partial(self.dynamics, z=z)
