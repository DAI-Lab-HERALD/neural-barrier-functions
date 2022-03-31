import torch
from torch import nn
import torch.nn.functional as F

from bound_propagation import IntervalBounds, LinearBounds

from bounds import bounds
from learned_cbf.partitioning import Partitions


class AdversarialNeuralSBF(nn.Module):
    def __init__(self, barrier, dynamics, factory, horizon):
        super().__init__()

        self.barrier = factory.build(barrier)
        self.barrier_dynamics = factory.build(nn.Sequential(dynamics, barrier))
        self.dynamics = dynamics

        self.horizon = horizon

        # self.rho = nn.Parameter(torch.empty((1,)))

        # TODO: Sample from state space (place on partitioning) and assert output is 1-D

    @property
    def alpha(self):
        """
        Parameterize alpha by rho to allow constraint free optimization.
        TODO: Change parameterization to allow [1, infty]
        :return: alpha = 1
        """
        return 1.0  # + F.softplus(self.rho)

    def beta(self, partitioning, **kwargs):
        """
        :return: beta in range [0, 1)
        """
        assert partitioning.safe is not None

        def reduce_mean(bounds):
            if isinstance(bounds, IntervalBounds):
                return IntervalBounds(
                    bounds.region,
                    bounds.lower.mean(dim=0) if bounds.lower is not None else None,
                    bounds.upper.mean(dim=0) if bounds.upper is not None else None
                )
            elif isinstance(bounds, LinearBounds):
                return LinearBounds(
                    bounds.region,
                    (bounds.lower[0].mean(dim=0), bounds.lower[1].mean(dim=0)) if bounds.lower is not None else None,
                    (bounds.upper[0].mean(dim=0), bounds.upper[1].mean(dim=0)) if bounds.upper is not None else None
                )
            else:
                raise ValueError('Bounds can only be linear or interval')

        if kwargs.get('method') == 'combined':
            kwargs.pop('method')
            # with torch.no_grad():
            #     _, upper = bounds(self.barrier_dynamics, partitioning.safe, bound_lower=False, method='crown_ibp_interval', reduce=reduce_mean, **kwargs)
            #     lower, _ = bounds(self.barrier, partitioning.safe, bound_upper=False, method='ibp', **kwargs)
            #
            #     expectation_no_beta = (upper.mean(dim=0) - lower / self.alpha).partition_max().clamp(min=0)
            #     idx = expectation_no_beta.argmax()
            #
            #     beta_max_partition = partitioning.safe[idx]
            #     beta_max_partition = Partitions((
            #         beta_max_partition.lower.unsqueeze(0),
            #         beta_max_partition.upper.unsqueeze(0)
            #     ))
            #
            # _, upper = bounds(self.barrier_dynamics, beta_max_partition, bound_lower=False, method='crown_ibp_interval', reduce=reduce_mean, **kwargs)
            # lower, _ = bounds(self.barrier, beta_max_partition, bound_upper=False, method='ibp', **kwargs)

            _, upper = bounds(self.barrier_dynamics, partitioning.safe, bound_lower=False, method='crown_ibp_interval', reduce=reduce_mean, **kwargs)
            lower, _ = bounds(self.barrier, partitioning.safe, bound_upper=False, method='ibp', **kwargs)
        else:
            # with torch.no_grad():
            #     _, upper = bounds(self.barrier_dynamics, partitioning.safe, bound_lower=False, reduce=reduce_mean, **kwargs)
            #     lower, _ = bounds(self.barrier, partitioning.safe, bound_upper=False, **kwargs)
            #
            #     expectation_no_beta = (upper - lower / self.alpha).partition_max().clamp(min=0)
            #     idx = expectation_no_beta.argmax()
            #
            #     beta_max_partition = partitioning.safe[idx]
            #     beta_max_partition = Partitions((
            #         beta_max_partition.lower.unsqueeze(0),
            #         beta_max_partition.upper.unsqueeze(0)
            #     ))
            #
            # _, upper = bounds(self.barrier_dynamics, beta_max_partition, bound_lower=False, reduce=reduce_mean, **kwargs)
            # lower, _ = bounds(self.barrier, beta_max_partition, bound_upper=False, **kwargs)

            _, upper = bounds(self.barrier_dynamics, partitioning.safe, bound_lower=False, reduce=reduce_mean, **kwargs)
            lower, _ = bounds(self.barrier, partitioning.safe, bound_upper=False, **kwargs)

        beta = (upper - lower / self.alpha).partition_max().view(-1)
        T = 0.02
        return torch.dot(F.softmax(beta / T, dim=0), beta.clamp(min=0))
        # return beta.max()
        # return beta.sum()

        # return torch.dot(beta.view(-1), partitioning.safe.volumes) / partitioning.safe.volume

    def gamma(self, partitioning, **kwargs):
        """
        :return: gamma in range [0, 1)
        """
        assert partitioning.initial is not None

        if kwargs.get('method') == 'combined':
            kwargs['method'] = 'crown_ibp_interval'

        # with torch.no_grad():
        #     _, upper = bounds(self.barrier, partitioning.initial, bound_lower=False, **kwargs)
        #     idx = upper.partition_max().argmax()
        #
        #     gamma_max_partition = partitioning.initial[idx]
        #     gamma_max_partition = Partitions((
        #         gamma_max_partition.lower.unsqueeze(0),
        #         gamma_max_partition.upper.unsqueeze(0)
        #     ))
        #
        # _, upper = bounds(self.barrier, gamma_max_partition, bound_lower=False, **kwargs)
        _, upper = bounds(self.barrier, partitioning.initial, bound_lower=False, **kwargs)
        gamma = upper.partition_max().view(-1)
        T = 0.02
        return torch.dot(F.softmax(gamma / T, dim=0), gamma.clamp(min=0))
        # return gamma.max()
        # return gamma.sum()

        # return torch.dot(gamma.view(-1), partitioning.initial.volumes) / partitioning.initial.volume

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

        if kwargs.get('method') == 'combined':
            kwargs['method'] = 'ibp'

        lower, _ = bounds(self.barrier, partitioning.unsafe, bound_upper=False, **kwargs)
        violation = (1 - lower).partition_max().clamp(min=0)
        # return violation.max()
        # return violation.sum()

        return torch.dot(violation.view(-1), partitioning.unsafe.volumes) / partitioning.unsafe.volume

    def loss_state_space(self, partitioning, **kwargs):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        if kwargs.get('method') == 'combined':
            kwargs['method'] = 'ibp'

        if partitioning.state_space is not None:
            lower, _ = bounds(self.barrier, partitioning.state_space, bound_upper=False, **kwargs)
            violation = (0 - lower).partition_max().clamp(min=0)
            # return violation.max()
            # return violation.sum()

            return torch.dot(violation.view(-1), partitioning.state_space.volumes) / partitioning.unsafe.volume
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


class EmpiricalNeuralSBF(nn.Module):
    def __init__(self, barrier, dynamics, horizon):
        super().__init__()

        self.barrier = barrier
        self.dynamics = dynamics

        self.horizon = horizon

        # self.rho = nn.Parameter(torch.empty((1,)))

        # TODO: Sample from state space (place on partitioning) and assert output is 1-D

    @property
    def alpha(self):
        """
        Parameterize alpha by rho to allow constraint free optimization.
        TODO: Change parameterization to allow [1, infty]
        :return: alpha = 1
        """
        return 1.0  # + F.softplus(self.rho)

    def beta(self, partitioning):
        """
        :return: beta in range [0, 1)
        """
        assert torch.all(self.dynamics.state_space(partitioning.safe.center, eps=partitioning.safe.width / 2))
        assert torch.any(self.dynamics.safe(partitioning.safe.center, eps=partitioning.safe.width / 2))

        x = partitioning.safe.center

        expectation = self.barrier(self.dynamics(x)).mean(dim=0)
        bx = self.barrier(x)

        beta = (expectation - bx / self.alpha)[:, 0].view(-1)
        T = 0.02
        return torch.dot(F.softmax(beta / T, dim=0), beta.clamp(min=0))
        # return beta.max()

    def gamma(self, partitioning):
        """
        :return: gamma in range [0, 1)
        """
        assert torch.all(self.dynamics.state_space(partitioning.initial.center, eps=partitioning.initial.width / 2))
        assert torch.any(self.dynamics.initial(partitioning.initial.center, eps=partitioning.initial.width / 2))

        x = partitioning.initial.center

        gamma = self.barrier(x)[:, 0].view(-1)
        T = 0.02
        return torch.dot(F.softmax(gamma / T, dim=0), gamma.clamp(min=0))
        # return gamma.max()

    def loss(self, partitioning, safety_weight=0.5):
        if safety_weight == 1.0:
            return self.loss_safety_prob(partitioning)
        elif safety_weight == 0.0:
            return self.loss_barrier(partitioning)

        loss_barrier = self.loss_barrier(partitioning)
        loss_safety_prob = self.loss_safety_prob(partitioning)

        return (1.0 - safety_weight) * loss_barrier + safety_weight * loss_safety_prob

    def loss_barrier(self, partitioning):
        loss = self.loss_state_space(partitioning) + self.loss_unsafe(partitioning)
        return loss

    def loss_unsafe(self, partitioning):
        """
        Ensure that B(x) >= 1 for all x in X_u.
        :return: Loss for unsafe set
        """
        assert torch.all(self.dynamics.state_space(partitioning.unsafe.center, eps=partitioning.unsafe.width / 2))
        assert torch.any(self.dynamics.unsafe(partitioning.unsafe.center, eps=partitioning.unsafe.width / 2))

        x = partitioning.unsafe.center

        violation = (1 - self.barrier(x)).clamp(min=0)
        return violation.mean()

    def loss_state_space(self, partitioning):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        assert torch.all(self.dynamics.state_space(partitioning.state_space.center))

        violation = (0 - self.barrier(partitioning.state_space.center)).clamp(min=0)
        return violation.mean()

    def loss_safety_prob(self, partitioning):
        """
        gamma + beta * horizon is an upper bound for probability of B(x) >= 1 in horizon steps.
        But we need to account for the fact that we backprop through dynamics in beta, num_samples times.
        :return: Upper bound for (1 - safety probability) adjusted to construct loss.
        """
        return self.gamma(partitioning) + self.beta(partitioning) * self.horizon
