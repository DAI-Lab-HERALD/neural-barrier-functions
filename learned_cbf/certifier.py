import torch
from bound_propagation import IntervalBounds, LinearBounds
from torch import nn

from bounds import bounds
from learned_cbf.partitioning import Partitions


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
        bx_violation_prob = gamma + beta * self.horizon

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


class SplittingNeuralSBFCertifier(nn.Module):
    def __init__(self, barrier, dynamics, factory, partitioning, horizon,
                 certification_threshold=1.0e-10, split_gap_stop_treshold=1e-8,
                 max_set_size=10000):
        super().__init__()

        self.barrier = factory.build(barrier)
        self.barrier_dynamics = factory.build(nn.Sequential(dynamics, barrier))
        self.dynamics = dynamics

        # Assumptions:
        # 1. Initial set, unsafe set, and safe set are partitioned.
        # 2. State space is optionally partitioned. If not, it should end with ReLU.
        # 3. Partitions containing the boundary of the safe / unsafe set belong to both (to ensure correctness)
        # 4. Partitions are hyperrectangular and non-overlapping
        self.initial_partitioning = partitioning

        self.horizon = horizon
        self.certification_threshold = certification_threshold
        self.split_gap_stop_treshold = split_gap_stop_treshold
        self.max_set_size = max_set_size

    @torch.no_grad()
    def beta(self, **kwargs):
        assert self.initial_partitioning.safe is not None
        set = self.initial_partitioning.safe

        min, max = self.min_max_beta(set, **kwargs)
        last_gap = torch.finfo(min.dtype).max

        while not self.should_stop_beta_gamma(set, min, max, last_gap):
            last_gap = (max - min).max().item()

            set = self.prune_beta_gamma(set, min, max)
            set = self.split_beta(set, **kwargs)
            set = self.region_prune(set, self.dynamics.safe)

            min, max = self.min_max_beta(set, **kwargs)

        return max.max().clamp(min=0)

    def min_max_beta(self, set, **kwargs):
        if kwargs.get('method') == 'optimal':
            kwargs.pop('method')

            expectation_lower_ibp, expectation_upper_ibp = bounds(self.barrier_dynamics, set, method='ibp', reduce=reduce_mean, **kwargs)
            expectation_lower_crown, expectation_upper_crown = bounds(self.barrier_dynamics, set, method='crown_ibp_linear', reduce=reduce_mean, **kwargs)

            lower_ibp, upper_ibp = bounds(self.barrier, set, method='ibp', **kwargs)
            lower_crown, upper_crown = bounds(self.barrier, set, method='crown_ibp_linear', **kwargs)

            beta_ibp_ibp = (expectation_upper_ibp - lower_ibp).partition_max()
            beta_ibp_crown = (expectation_upper_ibp - lower_crown).partition_max()
            beta_crown_ibp = (expectation_upper_crown - lower_ibp).partition_max()
            beta_crown_crown = (expectation_upper_crown - lower_crown).partition_max()
            max = torch.stack([beta_ibp_ibp, beta_ibp_crown, beta_crown_ibp, beta_crown_crown]).min(dim=0).values

            beta_ibp_ibp = (expectation_lower_ibp - upper_ibp).partition_min()
            beta_ibp_crown = (expectation_lower_ibp - upper_crown).partition_min()
            beta_crown_ibp = (expectation_lower_crown - upper_ibp).partition_min()
            beta_crown_crown = (expectation_lower_crown - upper_crown).partition_min()
            min = torch.stack([beta_ibp_ibp, beta_ibp_crown, beta_crown_ibp, beta_crown_crown]).max(dim=0).values
        else:
            expectation_lower, expectation_upper = bounds(self.barrier_dynamics, set, reduce=reduce_mean, **kwargs)
            lower, upper = bounds(self.barrier, set, **kwargs)

            max = (expectation_upper - lower).partition_max()
            min = (expectation_lower - upper).partition_min()

        return min.view(-1), max.view(-1)

    def split_beta(self, set, **kwargs):
        kwargs.pop('method', None)

        expectation_lower, expectation_upper = bounds(self.barrier_dynamics, set, method='crown_ibp_linear', reduce=reduce_mean, **kwargs)
        lower, upper = bounds(self.barrier, set, method='crown_ibp_linear', **kwargs)
        upperA = expectation_upper.A - lower.A
        lowerA = expectation_lower.A - upper.A

        split_dim = ((upperA.abs() + lowerA.abs())[:, 0] * set.width).argmax(dim=-1)
        partition_indices = torch.arange(0, set.lower.size(0), device=set.lower.device)
        split_dim = (partition_indices, split_dim)

        lower, upper = set.lower, set.upper
        mid = set.center

        p1_lower = lower.clone()
        p1_upper = upper.clone()
        p1_upper[split_dim] = mid[split_dim]

        p2_lower = lower.clone()
        p2_upper = upper.clone()
        p2_lower[split_dim] = mid[split_dim]

        lower, upper = torch.cat([p1_lower, p2_lower]), torch.cat([p1_upper, p2_upper])

        assert torch.all(lower <= upper + 1e-8)

        return Partitions((lower, upper))

    def region_prune(self, set, contain_func):
        center = set.center
        epsilon = set.width / 2

        overlaps = contain_func(center, epsilon)

        return Partitions((set.lower[overlaps], set.upper[overlaps]))

    @torch.no_grad()
    def gamma(self, **kwargs):
        assert self.initial_partitioning.initial is not None
        set = self.initial_partitioning.initial

        min, max = self.min_max(set, **kwargs)
        last_gap = torch.finfo(min.dtype).max

        while not self.should_stop_beta_gamma(set, min, max, last_gap):
            last_gap = (max - min).max().item()

            set = self.prune_beta_gamma(set, min, max)
            set = self.split(set, **kwargs)
            set = self.region_prune(set, self.dynamics.initial)

            min, max = self.min_max(set, **kwargs)

        return max.max().clamp(min=0)

    def prune_beta_gamma(self, set, min, max):
        largest_lower_bound = min.max()

        prune = (max <= 0.0) | (max <= largest_lower_bound)
        keep = ~prune

        if torch.all(prune):
            keep[max.argmax()] = True

        return Partitions((set.lower[keep], set.upper[keep]))

    def should_stop_beta_gamma(self, set, min, max, last_gap):
        gap = (max - min).max().item()
        abs_max = max.max().item()
        return len(set) > self.max_set_size or \
               abs_max <= 0.0 or \
               gap <= self.split_gap_stop_treshold or \
               gap >= last_gap - 1e-10

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
        assert self.initial_partitioning.unsafe is not None
        return self.violation(self.initial_partitioning.unsafe, 1, self.dynamics.unsafe, **kwargs)

    @torch.no_grad()
    def state_space_violation(self, **kwargs):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        if self.initial_partitioning.state_space is not None:
            return self.violation(self.initial_partitioning.state_space, 0, self.dynamics.state_space, **kwargs)
        else:
            # Assume that dynamics ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
            return 0.0

    @torch.no_grad()
    def violation(self, set, lower_bound, contain_func, **kwargs):
        min, max = self.min_max(set, **kwargs)
        last_gap = torch.finfo(min.dtype).max

        while not self.should_stop_violation(set, min, max, lower_bound, last_gap):
            last_gap = (max - min).max().item()

            set = self.prune_violation(set, min, max, lower_bound)
            set = self.split(set, **kwargs)
            set = self.region_prune(set, contain_func)

            min, max = self.min_max(set, **kwargs)

        return (lower_bound - min.min()).clamp(min=0)

    def min_max(self, set, **kwargs):
        if kwargs.get('method') == 'optimal':
            kwargs.pop('method')
            lower, upper = bounds(self.barrier, set, method='ibp', **kwargs)
            min_ibp = lower.partition_min()
            max_ibp = upper.partition_max()

            lower, upper = bounds(self.barrier, set, method='crown_ibp_interval', **kwargs)
            min_crown = lower.partition_min()
            max_crown = upper.partition_max()

            min = torch.max(min_ibp, min_crown)
            max = torch.min(max_ibp, max_crown)
        else:
            lower, upper = bounds(self.barrier, set, **kwargs)
            min = lower.partition_min()
            max = upper.partition_max()

        return min.view(-1), max.view(-1)

    def prune_violation(self, set, min, max, lower_bound):
        least_upper_bound = max.min()

        prune = (min >= lower_bound) | (min >= least_upper_bound)
        keep = ~prune

        if torch.all(prune):
            keep[max.argmax()] = True

        return Partitions((set.lower[keep], set.upper[keep]))

    def split(self, set, **kwargs):
        kwargs.pop('method', None)

        lower, upper = bounds(self.barrier, set, method='crown_ibp_linear', **kwargs)

        split_dim = ((upper.A.abs() + lower.A.abs())[:, 0] * set.width).argmax(dim=-1)
        partition_indices = torch.arange(0, set.lower.size(0), device=set.lower.device)
        split_dim = (partition_indices, split_dim)

        lower, upper = set.lower, set.upper
        mid = set.center

        p1_lower = lower.clone()
        p1_upper = upper.clone()
        p1_upper[split_dim] = mid[split_dim]

        p2_lower = lower.clone()
        p2_upper = upper.clone()
        p2_lower[split_dim] = mid[split_dim]

        lower, upper = torch.cat([p1_lower, p2_lower]), torch.cat([p1_upper, p2_upper])

        assert torch.all(lower <= upper + 1e-8)

        return Partitions((lower, upper))

    def should_stop_violation(self, set, min, max, lower_bound, last_gap):
        gap = (max - min).max().item()
        abs_min = min.min().item()
        return len(set) > self.max_set_size or \
               abs_min >= lower_bound or \
               gap <= self.split_gap_stop_treshold or \
               gap >= last_gap - 1e-10

    @torch.no_grad()
    def unsafety_prob(self, return_beta_gamma=False, **kwargs):
        """
        gamma + beta * horizon is an upper bound for probability of B(x) >= 1 in horizon steps.
        If alpha > 1.0, we can do better but gamma + beta * horizon remains an upper bound.
        :return: Upper bound for (1 - safety probability).
        """
        beta = self.beta(**kwargs)
        gamma = self.gamma(**kwargs)
        bx_violation_prob = gamma + beta * self.horizon

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
