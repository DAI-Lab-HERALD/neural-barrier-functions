import math

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


def overlap_rectangle(partition_lower, partition_upper, rect_lower, rect_upper):
    # Separating axis theorem
    return partition_upper[..., 0] >= rect_lower[0] and partition_lower[..., 0] <= rect_upper[0] and \
           partition_upper[..., 1] >= rect_lower[1] and partition_lower[..., 1] <= rect_upper[1]


def overlap_circle(partition_lower, partition_upper, center, radius):
    closest_point = torch.max(partition_lower, torch.min(partition_upper, center))
    distance = (closest_point - center).norm(dim=-1)
    return distance <= radius


def overlap_outside_rectangle(partition_lower, partition_upper, rect_lower, rect_upper):
    return partition_upper[..., 0] >= rect_upper[0] or partition_lower[..., 0] <= rect_lower[0] or \
           partition_upper[..., 1] >= rect_upper[1] or partition_lower[..., 1] <= rect_lower[1]


def overlap_outside_circle(partition_lower, partition_upper, center, radius):
    farthest_point = torch.where((partition_lower - center).abs() > (partition_upper - center).abs(), partition_lower, partition_upper)
    distance = (farthest_point - center).norm(dim=-1)
    return distance >= radius


class Polynomial(StochasticDynamics):
    def __init__(self, dynamics_config):
        super().__init__(
            PolynomialStep(dynamics_config),
            num_samples=dynamics_config['num_samples']
        )

    def initial(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_circle(lower_x, upper_x, torch.tensor([1.5, 0]), math.sqrt(0.25))
        cond2 = overlap_rectangle(lower_x, upper_x, torch.tensor([-1.8, -0.1]), torch.tensor([-1.2, 0.1]))
        cond3 = overlap_rectangle(lower_x, upper_x, torch.tensor([-1.4, -0.5]), torch.tensor([-1.2, 0.1]))

        return cond1 | cond2 | cond3

    def safe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_outside_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0]), math.sqrt(0.16))
        cond2 = overlap_outside_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.6, 0.5]))
        cond3 = overlap_outside_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.8, 0.3]))

        return cond1 & cond2 & cond3

    def unsafe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        cond1 = overlap_circle(lower_x, upper_x, torch.tensor([-1.0, -1.0]), math.sqrt(0.16))
        cond2 = overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.6, 0.5]))
        cond3 = overlap_rectangle(lower_x, upper_x, torch.tensor([0.4, 0.1]), torch.tensor([0.8, 0.3]))

        return cond1 | cond2 | cond3

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps
        else:
            lower_x, upper_x = x, x

        return (upper_x[..., 0] >= -3.5) & (lower_x[..., 0] <= 2.0) &\
               (upper_x[..., 1] >= -2.0) & (lower_x[..., 1] <= 1.0)

