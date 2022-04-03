import torch
from bound_propagation import IntervalBounds, LinearBounds
from torch import nn

from bounds import bounds
from learned_cbf.partitioning import Partitions


class NeuralSBFCertifier(nn.Module):
    def __init__(self, barrier, dynamics, factory, partitioning, horizon, certification_threshold=1.0e-10):
        super().__init__()

        self.barrier = factory.build(barrier)
        self.barrier_dynamics = factory.build(nn.Sequential(dynamics, barrier))
        self.dynamics = dynamics

        # Assumptions:
        # 1. Initial set, unsafe set, and safe set are partitioned.
        # 2. State space is optionally partitioned. If not, it should end with ReLU.
        # 3. Partitions containing the boundary of the safe / unsafe set belong to both (to ensure correctness)
        # 4. Partitions are hyperrectangular and non-overlapping
        self.partitioning = partitioning

        self.horizon = horizon
        self.certification_threshold = certification_threshold

    @torch.no_grad()
    def beta(self, **kwargs):
        """
        :return: beta in range [0, 1)
        """
        assert self.partitioning.safe is not None

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

        if kwargs.get('method') == 'optimal':
            kwargs.pop('method')

            _, upper_ibp = bounds(self.barrier_dynamics, self.partitioning.safe, bound_lower=False, method='ibp', reduce=reduce_mean, **kwargs)
            _, upper_crown = bounds(self.barrier_dynamics, self.partitioning.safe, bound_lower=False, method='crown_ibp_linear', reduce=reduce_mean, **kwargs)

            lower_ibp, _ = bounds(self.barrier, self.partitioning.safe, bound_upper=False, method='ibp', **kwargs)
            lower_crown, _ = bounds(self.barrier, self.partitioning.safe, bound_upper=False, method='crown_ibp_linear', **kwargs)

            beta_ibp_ibp = (upper_ibp - lower_ibp).partition_max().max().clamp(min=0)
            beta_ibp_crown = (upper_ibp - lower_crown).partition_max().max().clamp(min=0)
            beta_crown_ibp = (upper_crown - lower_ibp).partition_max().max().clamp(min=0)
            beta_crown_crown = (upper_crown - lower_crown).partition_max().max().clamp(min=0)

            beta = torch.min(torch.stack([beta_ibp_ibp, beta_ibp_crown, beta_crown_ibp, beta_crown_crown]))
        else:
            _, upper = bounds(self.barrier_dynamics, self.partitioning.safe, bound_lower=False, reduce=reduce_mean, **kwargs)
            lower, _ = bounds(self.barrier, self.partitioning.safe, bound_upper=False, **kwargs)

            beta = (upper - lower).partition_max().max().clamp(min=0)

        return beta

    @torch.no_grad()
    def gamma(self, **kwargs):
        """
        :return: gamma in range [0, 1)
        """
        assert self.partitioning.initial is not None

        if kwargs.get('method') == 'optimal':
            kwargs.pop('method')

            _, upper_ibp = bounds(self.barrier, self.partitioning.initial, bound_lower=False, method='ibp', **kwargs)
            gamma_ibp = upper_ibp.partition_max().max().clamp(min=0)
            _, upper_crown = bounds(self.barrier, self.partitioning.initial, bound_lower=False, method='crown_ibp_interval', **kwargs)
            gamma_crown = upper_crown.partition_max().max().clamp(min=0)

            gamma = torch.min(gamma_ibp, gamma_crown)
        else:
            _, upper = bounds(self.barrier, self.partitioning.initial, bound_lower=False, **kwargs)
            gamma = upper.partition_max().max().clamp(min=0)

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

        return self.violation(self.partitioning.unsafe, 1, self.partitioning.unsafe.volumes, **kwargs)

    @torch.no_grad()
    def state_space_violation(self, **kwargs):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        if self.partitioning.state_space is not None:
            return self.violation(self.partitioning.state_space, 0, self.partitioning.state_space.volumes, **kwargs)
        else:
            # Assume that dynamics ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
            return 0.0

    @torch.no_grad()
    def violation(self, set, lower_bound, volumes, **kwargs):
        if kwargs.get('method') == 'optimal':
            kwargs.pop('method')
            lower, _ = bounds(self.barrier, set, bound_upper=False, method='ibp', **kwargs)
            violation_ibp = (lower_bound - lower).partition_max().clamp(min=0)

            lower, _ = bounds(self.barrier, set, bound_upper=False, method='crown_ibp_interval', **kwargs)
            violation_crown = (lower_bound - lower).partition_max().clamp(min=0)

            violation = torch.min(violation_ibp, violation_crown)
        else:
            lower, _ = bounds(self.barrier, set, bound_upper=False, **kwargs)
            violation = (lower_bound - lower).partition_max().clamp(min=0)

        return torch.dot(violation.view(-1), volumes) / volumes.sum()

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
        return self.barrier_violation(**kwargs).item() <= self.certification_threshold
