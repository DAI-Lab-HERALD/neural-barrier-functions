from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from .partitioning import Partitions
from .networks import BetaNetwork
from .bounds import bounds


class AdversarialNeuralSBF(nn.Module):
    def __init__(self, barrier, dynamics, factory, horizon, T=0.05):
        super().__init__()

        self.barrier = factory.build(barrier)
        self.beta_network = factory.build(BetaNetwork(dynamics, barrier))
        self.dynamics = dynamics

        self.horizon = horizon
        self.T = T

        # TODO: Sample from state space (place on partitioning) and assert output is 1-D

    def beta(self, partitioning, **kwargs):
        """
        :return: beta in range [0, 1)
        """
        assert partitioning.safe is not None

        _, upper = bounds(self.beta_network, partitioning.safe, bound_lower=False, **kwargs)
        beta = upper.partition_max().view(-1)

        # return torch.dot(beta.clamp(min=0), partitioning.safe.volumes) / partitioning.safe.volume
        # return torch.dot(torch.softmax(beta / self.T, 0), beta.clamp(min=0))
        return beta.clamp(min=0.0).max()

    def gamma(self, partitioning, **kwargs):
        """
        :return: gamma in range [0, 1)
        """
        assert partitioning.initial is not None

        _, upper = bounds(self.barrier, partitioning.initial, bound_lower=False, **kwargs)
        gamma = upper.partition_max().view(-1)

        # return torch.dot(gamma.clamp(min=0), partitioning.initial.volumes) / partitioning.initial.volume
        # return torch.dot(torch.softmax(gamma / self.T, 0), gamma.clamp(min=0))
        return gamma.clamp(min=0.0).max()

    def loss(self, partitioning, safety_weight=0.5, **kwargs):
        if safety_weight == 1.0:
            return self.loss_safety_prob(partitioning, **kwargs)
        elif safety_weight == 0.0:
            return self.loss_barrier(partitioning, **kwargs)

        loss_barrier = self.loss_barrier(partitioning, **kwargs)
        loss_safety_prob = self.loss_safety_prob(partitioning, **kwargs)

        return (1.0 - safety_weight) * loss_barrier + safety_weight * loss_safety_prob

    def loss_barrier(self, partitioning, violation_normalization_factor=1.0, **kwargs):
        loss = self.loss_state_space(partitioning, **kwargs) + self.loss_unsafe(partitioning, **kwargs)
        return loss * violation_normalization_factor

    def loss_unsafe(self, partitioning, **kwargs):
        """
        Ensure that B(x) >= 1 for all x in X_u.
        :return: Loss for unsafe set
        """
        assert partitioning.unsafe is not None

        lower, _ = bounds(self.barrier, partitioning.unsafe, bound_upper=False, **kwargs)
        violation = ((1.0 + 1e-8) - lower).partition_max().view(-1)

        # return torch.dot(torch.softmax(violation / self.T, 0), violation.clamp(min=0))
        return torch.dot(violation.clamp(min=0.0), partitioning.unsafe.volumes) / partitioning.unsafe.volume
        # return violation.clamp(min=0).max()

    def loss_state_space(self, partitioning, **kwargs):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        if partitioning.state_space is not None:
            lower, _ = bounds(self.barrier, partitioning.state_space, bound_upper=False, **kwargs)
            violation = (1e-8 - lower).partition_max().view(-1)

            # return torch.dot(torch.softmax(violation / self.T, 0), violation.clamp(min=0))
            return torch.dot(violation.clamp(min=0.0), partitioning.state_space.volumes) / partitioning.unsafe.volume
            # return violation.clamp(min=0).max()
        else:
            # Assume that dynamics ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
            return 0.0

    def loss_safety_prob(self, partitioning, **kwargs):
        """
        gamma + beta * horizon is an upper bound for probability of B(x) >= 1 in horizon steps.
        But we need to account for the fact that we backprop through dynamics in beta, num_samples times.
        :return: Upper bound for (1 - safety probability) adjusted to construct loss.
        """
        loss = self.gamma(partitioning, **kwargs) + self.beta(partitioning, **kwargs) * self.horizon
        return loss

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        self.barrier.load_state_dict(OrderedDict([(key.replace('barrier.', ''), value) for key, value in state_dict.items() if key.startswith('barrier.')]))
        self.beta_network.load_state_dict(OrderedDict([(key.replace('beta_network.', ''), value) for key, value in state_dict.items() if key.startswith('beta_network.')]))


class EmpiricalNeuralSBF(nn.Module):
    def __init__(self, barrier, dynamics, horizon):
        super().__init__()

        self.barrier = barrier
        self.dynamics = dynamics

        self.horizon = horizon

        # TODO: Sample from state space (place on partitioning) and assert output is 1-D

    def beta(self, partitioning):
        """
        :return: beta in range [0, 1)
        """
        assert torch.all(self.dynamics.state_space(partitioning.safe.center, eps=partitioning.safe.width / 2))
        assert torch.any(self.dynamics.safe(partitioning.safe.center, eps=partitioning.safe.width / 2))

        x = partitioning.safe.center

        expectation = self.barrier(self.dynamics(x)).mean(dim=0)
        bx = self.barrier(x)

        beta = (expectation - bx).view(-1)
        return beta.clamp(min=0.0).max()

    def gamma(self, partitioning):
        """
        :return: gamma in range [0, 1)
        """
        assert torch.all(self.dynamics.state_space(partitioning.initial.center, eps=partitioning.initial.width / 2))
        assert torch.any(self.dynamics.initial(partitioning.initial.center, eps=partitioning.initial.width / 2))

        x = partitioning.initial.center

        gamma = self.barrier(x).view(-1)
        return gamma.clamp(min=0.0).max()

    def loss(self, partitioning, safety_weight=0.5, **kwargs):
        if safety_weight == 1.0:
            return self.loss_safety_prob(partitioning)
        elif safety_weight == 0.0:
            return self.loss_barrier(partitioning, **kwargs)

        loss_barrier = self.loss_barrier(partitioning, **kwargs)
        loss_safety_prob = self.loss_safety_prob(partitioning)

        return (1.0 - safety_weight) * loss_barrier + safety_weight * loss_safety_prob

    def loss_barrier(self, partitioning, violation_normalization_factor=1.0, **kwargs):
        loss = self.loss_state_space(partitioning) + self.loss_unsafe(partitioning)
        return loss * violation_normalization_factor

    def loss_unsafe(self, partitioning):
        """
        Ensure that B(x) >= 1 for all x in X_u.
        :return: Loss for unsafe set
        """
        assert torch.all(self.dynamics.state_space(partitioning.unsafe.center, eps=partitioning.unsafe.width / 2))
        assert torch.any(self.dynamics.unsafe(partitioning.unsafe.center, eps=partitioning.unsafe.width / 2))

        x = partitioning.unsafe.center

        violation = (1.0 - self.barrier(x)).clamp(min=0.0)
        return violation.mean()

    def loss_state_space(self, partitioning):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        assert torch.all(self.dynamics.state_space(partitioning.state_space.center))

        violation = (0.0 - self.barrier(partitioning.state_space.center)).clamp(min=0.0)
        return violation.mean()

    def loss_safety_prob(self, partitioning):
        """
        gamma + beta * horizon is an upper bound for probability of B(x) >= 1 in horizon steps.
        But we need to account for the fact that we backprop through dynamics in beta, num_samples times.
        :return: Upper bound for (1 - safety probability) adjusted to construct loss.
        """
        return self.gamma(partitioning) + self.beta(partitioning) * self.horizon
