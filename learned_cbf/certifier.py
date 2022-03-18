import torch
from torch import nn

from learned_cbf.partitioning import Partitions


class NeuralSBFCertifier(nn.Module):
    def __init__(self, barrier, dynamics, partitioning, horizon):
        super().__init__()

        self.barrier = barrier
        self.dynamics = dynamics

        # Assumptions:
        # 1. Initial set, unsafe set, and safe set are partitioned.
        # 2. State space is optionally partitioned. If not, it should end with ReLU.
        # 3. Partitions containing the boundary of the safe / unsafe set belong to both (to ensure correctness)
        # 4. Partitions are hyperrectangular and non-overlapping
        self.partitioning = partitioning

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

    @torch.no_grad()
    def beta(self, **kwargs):
        """
        :return: beta in range [0, 1)
        """
        assert self.partitioning.safe is not None

        if kwargs.get('method') == 'optimal':
            kwargs.pop('method')

            _, upper = self.barrier.bounds(self.partitioning.safe, self.dynamics, bound_lower=False, method='crown_ibp_linear', **kwargs)
            lower, _ = self.barrier.bounds(self.partitioning.safe, bound_upper=False, method='ibp', **kwargs)
        else:
            _, upper = self.barrier.bounds(self.partitioning.safe, self.dynamics, bound_lower=False, **kwargs)
            lower, _ = self.barrier.bounds(self.partitioning.safe, bound_upper=False, **kwargs)

        beta = (upper.mean(dim=0) - lower / self.alpha).partition_max().max().clamp(min=0)

        return beta

    @torch.no_grad()
    def gamma(self, **kwargs):
        """
        :return: gamma in range [0, 1)
        """
        assert self.partitioning.initial is not None

        if kwargs.get('method') == 'optimal':
            kwargs['method'] = 'crown_ibp_linear'

        _, upper = self.barrier.bounds(self.partitioning.initial, bound_lower=False, **kwargs)
        gamma = upper.partition_max().max()

        return gamma

    @torch.no_grad()
    def barrier_violation(self, **kwargs):
        loss = self.state_space_violation(**kwargs) + self.unsafe_violation(**kwargs)
        return loss

    @torch.no_grad()
    def unsafe_violation(self, **kwargs):
        """
        Ensure that B(x) >= 1 for all x in X_u.
        :return: Loss for unsafe set
        """
        assert self.partitioning.unsafe is not None

        if kwargs.get('method') == 'optimal':
            kwargs['method'] = 'ibp'

        if kwargs.get('method') == 'optimal':
            kwargs.pop('method')
            lower, _ = self.barrier.bounds(self.partitioning.unsafe, bound_upper=False, method='ibp', **kwargs)
            violation_ibp = (1 - lower).partition_max().clamp(min=0)

            lower, _ = self.barrier.bounds(self.partitioning.unsafe, bound_upper=False, method='crown_ibp_interval', **kwargs)
            violation_crown = (1 - lower).partition_max().clamp(min=0)

            violation = torch.min(violation_ibp, violation_crown).sum()
        else:
            lower, _ = self.barrier.bounds(self.partitioning.unsafe, bound_upper=False, **kwargs)
            violation = (1 - lower).partition_max().clamp(min=0).sum()

        return violation / self.partitioning.unsafe.volume

    @torch.no_grad()
    def state_space_violation(self, **kwargs):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        if kwargs.get('method') == 'optimal':
            kwargs['method'] = 'ibp'

        if self.partitioning.state_space is not None:
            if kwargs.get('method') == 'optimal':
                kwargs.pop('method')
                lower, _ = self.barrier.bounds(self.partitioning.state_space, bound_upper=False, method='ibp', **kwargs)
                violation_ibp = (0 - lower).partition_max().clamp(min=0)

                lower, _ = self.barrier.bounds(self.partitioning.state_space, bound_upper=False, method='crown_ibp_interval', **kwargs)
                violation_crown = (0 - lower).partition_max().clamp(min=0)

                violation = torch.min(violation_ibp, violation_crown).sum()
            else:
                lower, _ = self.barrier.bounds(self.partitioning.state_space, bound_upper=False, **kwargs)
                violation = (0 - lower).partition_max().clamp(min=0).sum()

            return violation / self.partitioning.state_space.volume
        else:
            # Assume that dynamics ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
            return 0.0

    @torch.no_grad()
    def unsafety_prob(self, return_beta_gamma=False, **kwargs):
        """
        gamma + beta * horizon is an upper bound for probability of B(x) >= 1 in horizon steps.
        If alpha > 1.0, we can do better but gamma + beta * horizon remains an upper bound.
        :return: Upper bound for (1 - safety probability).
        """
        beta = self.beta(**kwargs)
        gamma = self.gamma(**kwargs)
        if self.alpha == 1.0:
            bx_violation_prob = gamma + beta * self.horizon
        elif self.alpha * beta / (1 - self.alpha) <= 1:
            bx_violation_prob = 1 - (1 - gamma) * beta ** self.horizon
        else:
            bx_violation_prob = gamma * self.alpha ** (-self.horizon) + \
                (1 - self.alpha ** (-self.horizon)) * self.alpha * beta / (self.alpha - 1)

        if return_beta_gamma:
            return bx_violation_prob, beta, gamma

        return bx_violation_prob

    @torch.no_grad()
    def certify(self, **kwargs):
        """
        Certify that a trained barrier network is a valid barrier using the barrier loss
        Allow a small violation to account for potential numerical (FP) errors.
        :return: true if the barrier network is a valid barrier
        """
        return self.unsafe_violation(**kwargs).item() <= 1.0e-10
