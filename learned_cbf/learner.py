import torch
from torch import nn

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

        if kwargs.get('method') == 'combined':
            kwargs.pop('method')
            with torch.no_grad():
                _, upper = bounds(self.barrier_dynamics, partitioning.safe, bound_lower=False, method='crown_ibp_linear', **kwargs)
                lower, _ = bounds(self.barrier, partitioning.safe, bound_upper=False, method='ibp', **kwargs)

                expectation_no_beta = (upper.mean(dim=0) - lower / self.alpha).partition_max().clamp(min=0)
                idx = expectation_no_beta.argmax()

                del lower, upper, expectation_no_beta

                beta_max_partition = partitioning.safe[idx]
                beta_max_partition = Partitions((
                    beta_max_partition.lower.unsqueeze(0),
                    beta_max_partition.upper.unsqueeze(0)
                ))

            _, upper = bounds(self.barrier_dynamics, beta_max_partition, bound_lower=False, method='crown_ibp_linear', **kwargs)
            lower, _ = bounds(self.barrier, beta_max_partition, bound_upper=False, method='ibp', **kwargs)
            beta = (upper.mean(dim=0) - lower / self.alpha).partition_max().clamp(min=0)
        else:
            with torch.no_grad():
                _, upper = bounds(self.barrier_dynamics, partitioning.safe, bound_lower=False, **kwargs)
                lower, _ = bounds(self.barrier, partitioning.safe, bound_upper=False, **kwargs)

                expectation_no_beta = (upper.mean(dim=0) - lower / self.alpha).partition_max().clamp(min=0)
                idx = expectation_no_beta.argmax()

                del lower, upper, expectation_no_beta

                beta_max_partition = partitioning.safe[idx]
                beta_max_partition = Partitions((
                    beta_max_partition.lower.unsqueeze(0),
                    beta_max_partition.upper.unsqueeze(0)
                ))

            _, upper = bounds(self.barrier_dynamics, beta_max_partition, bound_lower=False, **kwargs)
            lower, _ = bounds(self.barrier, beta_max_partition, bound_upper=False, **kwargs)
            beta = (upper.mean(dim=0) - lower / self.alpha).partition_max().clamp(min=0)

        return beta

    def gamma(self, partitioning, **kwargs):
        """
        :return: gamma in range [0, 1)
        """
        assert partitioning.initial is not None

        if kwargs.get('method') == 'combined':
            kwargs['method'] = 'crown_ibp_linear'

        with torch.no_grad():
            _, upper = bounds(self.barrier, partitioning.initial, bound_lower=False, **kwargs)
            idx = upper.partition_max().argmax()

            del upper

            gamma_max_partition = partitioning.initial[idx]
            gamma_max_partition = Partitions((
                gamma_max_partition.lower.unsqueeze(0),
                gamma_max_partition.upper.unsqueeze(0)
            ))

        _, upper = bounds(self.barrier, gamma_max_partition, bound_lower=False, **kwargs)
        gamma = upper.partition_max().clamp(min=0)

        return gamma

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

        return torch.dot(violation.view(-1), partitioning.unsafe.volumes)

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

            return torch.dot(violation.view(-1), partitioning.state_space.volumes)
        else:
            # Assume that dynamics ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
            return 0.0

    def loss_safety_prob(self, partitioning, **kwargs):
        """
        gamma + beta * horizon is an upper bound for probability of B(x) >= 1 in horizon steps.
        But we need to account for the fact that we backprop through dynamics in beta, num_samples times.
        :return: Upper bound for (1 - safety probability) adjusted to construct loss.
        """
        return self.gamma(partitioning, **kwargs) + self.beta(partitioning, **kwargs) * self.horizon


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

    def beta(self, x):
        """
        :return: beta in range [0, 1)
        """
        assert torch.all(self.dynamics.state_space(x))
        assert torch.any(self.dynamics.safe(x))

        x = x[self.dynamics.safe(x)]

        expectation = self.barrier(self.dynamics(x)).mean(dim=0)
        bx = self.barrier(x)

        return (expectation - bx / self.alpha).max().clamp(min=0)

    def gamma(self, x):
        """
        :return: gamma in range [0, 1)
        """
        assert torch.all(self.dynamics.state_space(x))
        assert torch.any(self.dynamics.initial(x))

        x = x[self.dynamics.initial(x)]

        return self.barrier(x).max().clamp(min=0)

    def loss(self, x, safety_weight=0.5):
        if safety_weight == 1.0:
            return self.loss_safety_prob(x)
        elif safety_weight == 0.0:
            return self.loss_barrier(x)

        loss_barrier = self.loss_barrier(x)
        loss_safety_prob = self.loss_safety_prob(x)

        return (1.0 - safety_weight) * loss_barrier + safety_weight * loss_safety_prob

    def loss_barrier(self, x):
        loss = self.loss_state_space(x) + self.loss_unsafe(x)
        return loss

    def loss_unsafe(self, x):
        """
        Ensure that B(x) >= 1 for all x in X_u.
        :return: Loss for unsafe set
        """
        assert torch.all(self.dynamics.state_space(x))
        assert torch.any(self.dynamics.unsafe(x))

        x = x[self.dynamics.unsafe(x)]

        violation = (1 - self.barrier(x)).clamp(min=0).mean()
        return violation

    def loss_state_space(self, x):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        assert torch.all(self.dynamics.state_space(x))

        violation = (0 - self.barrier(x)).clamp(min=0).mean()
        return violation

    def loss_safety_prob(self, x):
        """
        gamma + beta * horizon is an upper bound for probability of B(x) >= 1 in horizon steps.
        But we need to account for the fact that we backprop through dynamics in beta, num_samples times.
        :return: Upper bound for (1 - safety probability) adjusted to construct loss.
        """
        return self.gamma(x) + self.beta(x) * self.horizon
