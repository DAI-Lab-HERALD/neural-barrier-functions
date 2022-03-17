import torch
from torch import nn
from torch.distributions import Normal

from learned_cbf.dynamics import StochasticDynamics


class PolynomialStep(nn.Module):
    def __init__(self, dynamics_config):
        super().__init__()

        del self.weight
        del self.bias

        self.dt = dynamics_config['dt']

        dist = Normal(0.0, dynamics_config['sigma'])
        self.z = dist.sample((dynamics_config['num_samples'],)).view(-1, 1, 1)

    def forward(self, x):
        x1 = self.dt * x[..., 0] + self.z
        x2 = self.dt * ((x[..., 0] ** 3) / 3.0 - x[..., 0] - x[..., 1]).unsqueeze(0).expand_like(x1)

        x = torch.stack([x1, x2], dim=-1)
        return x

    # TODO: Define IBP and CROWN for this step


class Polynomial(StochasticDynamics):
    def __init__(self, dynamics_config):
        super().__init__(
            PolynomialStep(dynamics_config),
            num_samples=dynamics_config['num_samples']
        )

    def safe(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        cond1 = (x1 + 1.0) ** 2 + (x2 + 1) ** 2 <= 0.16
        cond2 = (x1 >= 0.4) & (x1 <= 0.6) & (x2 >= 0.1) & (x2 <= 0.5)
        cond3 = (x1 >= 0.4) & (x1 <= 0.8) & (x2 >= 0.1) & (x2 <= 0.3)

        return ~(cond1 | cond2 | cond3)

    def initial(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        cond1 = (x1 - 1.5) ** 2 + x2 ** 2 <= 0.25
        cond2 = (x1 >= -1.8) & (x1 <= -1.2) & (x2 >= -0.1) & (x2 <= 0.1)
        cond3 = (x1 >= -1.4) & (x1 <= -1.2) & (x2 >= -0.5) & (x2 <= 0.1)

        return cond1 | cond2 | cond3

    def state_space(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        return (x1 >= -3.5) & (x1 <= 2.0) & (x2 >= -2.0) & (x2 <= 1.0)

