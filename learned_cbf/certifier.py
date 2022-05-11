import logging
import math

import torch
from bound_propagation import Clamp, IntervalBounds, LinearBounds, HyperRectangle
from torch import nn

from .dynamics import AdditiveGaussianDynamics
from .bounds import bounds, Affine, LearnedCBFBoundModelFactory
from .partitioning import Partitions
from .networks import BetaNetwork, AdditiveGaussianBetaNetwork

logger = logging.getLogger(__name__)


class NeuralSBFCertifier(nn.Module):
    def __init__(self, barrier, dynamics, factory, partitioning, horizon, certification_threshold=1.0e-10):
        super().__init__()

        barrier = nn.Sequential(barrier, Clamp(max=1.0 + 1e-6))

        self.barrier = factory.build(barrier)
        self.beta_network = factory.build(BetaNetwork(dynamics, barrier))
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

            _, upper_ibp = bounds(self.beta_network, self.partitioning.safe, bound_lower=False, method='ibp', **kwargs)
            _, upper_crown = bounds(self.beta_network, self.partitioning.safe, bound_lower=False, method='crown_interval', **kwargs)

            beta_ibp = upper_ibp.partition_max().max().clamp(min=0)
            beta_crown = upper_crown.partition_max().max().clamp(min=0)

            beta = torch.min(beta_ibp, beta_crown)
        else:
            _, upper = bounds(self.beta_network, self.partitioning.safe, bound_lower=False, **kwargs)

            beta = upper.partition_max().max().clamp(min=0)

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
            _, upper_crown = bounds(self.barrier, self.partitioning.initial, bound_lower=False, method='crown_interval', **kwargs)
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

            lower, _ = bounds(self.barrier, set, bound_upper=False, method='crown_interval', **kwargs)
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
                 certification_threshold=1.0e-10, split_gap_stop_treshold=1e-6,
                 max_set_size=20000):
        super().__init__()

        barrier = nn.Sequential(barrier, Clamp(max=1.0 + 1e-6))

        self.barrier = factory.build(barrier)
        self.beta_network = factory.build(BetaNetwork(dynamics, barrier))
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
        last_gap = [torch.finfo(min.dtype).max for _ in range(10)]

        while not self.should_stop_beta_gamma('BETA', set, min, max, last_gap):
            last_gap.append((max.max() - min.max()).item())
            last_gap.pop(0)

            set, prune_all = self.prune_beta_gamma(set, min, max)

            if prune_all:
                logger.warning(f'Pruning all in beta: {min}, {max}, last gap: {last_gap[-1]}')
                break

            set = self.split_beta(set, **kwargs)
            set = self.region_prune(set, self.dynamics.safe)

            min, max = self.min_max_beta(set, **kwargs)

        return max.max().clamp(min=0)

    def min_max_beta(self, set, **kwargs):
        if kwargs.get('method') == 'optimal':
            kwargs.pop('method')

            lower_ibp, upper_ibp = bounds(self.beta_network, set, method='ibp', **kwargs)
            lower_crown, upper_crown = bounds(self.beta_network, set, method='crown_interval', **kwargs)

            beta_ibp = upper_ibp.partition_max()
            beta_crown = upper_crown.partition_max()
            max = torch.min(beta_ibp, beta_crown)

            beta_ibp = lower_ibp.partition_min()
            beta_crown = lower_crown.partition_min()
            min = torch.max(beta_ibp, beta_crown)
        else:
            lower, upper = bounds(self.beta_network, set, **kwargs)

            max = upper.partition_max()
            min = lower.partition_min()

        return min.view(-1), max.view(-1)

    def split_beta(self, set, **kwargs):
        kwargs.pop('method', None)

        lower, upper = bounds(self.beta_network, set, method='crown_linear', **kwargs)

        split_dim = ((lower.A.abs() + upper.A.abs())[:, 0] * set.width).argmax(dim=-1)
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
        last_gap = [torch.finfo(min.dtype).max for _ in range(10)]

        while not self.should_stop_beta_gamma('GAMMA', set, min, max, last_gap):
            last_gap.append((max.max() - min.max()).item())
            last_gap.pop(0)

            set, prune_all = self.prune_beta_gamma(set, min, max)

            if prune_all:
                logger.warning(f'Pruning all in gamma: {min}, {max}, last gap: {last_gap[-1]}')
                break

            set = self.split(set, **kwargs)
            set = self.region_prune(set, self.dynamics.initial)

            min, max = self.min_max(set, **kwargs)

        return max.max().clamp(min=0)

    def prune_beta_gamma(self, set, min, max):
        largest_lower_bound = min.max()

        prune = (max <= 0.0) | (max <= largest_lower_bound)
        keep = ~prune

        if torch.all(prune):
            return set, True

        return Partitions((set.lower[keep], set.upper[keep])), False

    def should_stop_beta_gamma(self, label, set, min, max, last_gap):
        gap = (max.max() - min.max()).item()
        abs_max = max.max().item()

        logger.debug(f'[{label}] Gap: {gap}, set size: {len(set)}, upper bound: {max.max().item()}')

        return len(set) > self.max_set_size or \
               abs_max <= 0.0 or \
               gap <= self.split_gap_stop_treshold or \
               gap >= torch.tensor(last_gap).max()

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
        return self.violation('UNSAFE', self.initial_partitioning.unsafe, 1, self.dynamics.unsafe, **kwargs)

    @torch.no_grad()
    def state_space_violation(self, **kwargs):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        if self.initial_partitioning.state_space is not None:
            return self.violation('STATE_SPACE', self.initial_partitioning.state_space, 0, self.dynamics.state_space, **kwargs)
        else:
            # Assume that dynamics ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
            return 0.0

    @torch.no_grad()
    def violation(self, label, set, lower_bound, contain_func, **kwargs):
        min, max = self.min_max(set, **kwargs)
        last_gap = [torch.finfo(min.dtype).max for _ in range(10)]

        while not self.should_stop_violation(label, set, min, max, lower_bound, last_gap):
            last_gap.append((max.min() - min.min()).item())
            last_gap.pop(0)

            set, prune_all = self.prune_violation(set, min, max, lower_bound)

            if prune_all:
                logger.warning(f'Pruning all in violation: {min}, {max}, last gap: {last_gap[-1]}')
                break

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

            lower, upper = bounds(self.barrier, set, method='crown_interval', **kwargs)
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
            return set, True

        return Partitions((set.lower[keep], set.upper[keep])), False

    def split(self, set, **kwargs):
        kwargs.pop('method', None)

        lower, upper = bounds(self.barrier, set, method='crown_linear', **kwargs)

        split_dim = ((lower.A.abs() + upper.A.abs())[:, 0] * set.width).argmax(dim=-1)
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

    def should_stop_violation(self, label, set, min, max, lower_bound, last_gap):
        gap = (max.min() - min.min()).item()
        abs_min = min.min().item()

        logger.debug(f'[{label}] Gap: {gap}, set size: {len(set)}, lower bound: {min.min().item()}')

        return len(set) > self.max_set_size or \
               abs_min >= lower_bound or \
               gap <= self.split_gap_stop_treshold or \
               gap >= torch.tensor(last_gap).max()

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


class AdditiveGaussianSplittingNeuralSBFCertifier(nn.Module):
    def __init__(self, barrier: nn.Module, dynamics: AdditiveGaussianDynamics, factory, initial_partitioning, horizon,
                 certification_threshold=1.0e-10, split_gap_stop_treshold=1e-6, max_set_size=50000, device=None):
        super().__init__()

        assert isinstance(dynamics, AdditiveGaussianDynamics)

        self.barrier = factory.build(barrier)

        loc, scale = dynamics.v
        loc, scale = loc.to(device), scale.to(device)
        factory.kwargs['state_space_bounds'] = initial_partitioning.state_space.lower[0], initial_partitioning.state_space.upper[0]
        factory.kwargs['slices'] = [200 for _ in range(initial_partitioning.state_space.lower.size(-1))]
        self.beta_network = factory.build(AdditiveGaussianBetaNetwork(barrier, dynamics.nominal_system, loc, scale))
        self.dynamics = dynamics

        # Assumptions:
        # 1. Initial set, unsafe set, and safe set are partitioned.
        # 2. State space is optionally partitioned. If not, it should end with ReLU.
        # 3. Partitions containing the boundary of the safe / unsafe set belong to both (to ensure correctness)
        # 4. Partitions are hyperrectangular and non-overlapping
        self.initial_partitioning = initial_partitioning

        self.horizon = horizon
        self.certification_threshold = certification_threshold
        self.split_gap_stop_treshold = split_gap_stop_treshold
        self.max_set_size = max_set_size

    @torch.no_grad()
    def beta(self, **kwargs):
        assert self.initial_partitioning.safe is not None

        set = self.initial_partitioning.safe

        kwargs['method'] = 'crown_linear'
        lower, upper = bounds(self.beta_network, set, **kwargs)
        min, max = self.min_max_beta(lower, upper)
        last_gap = [torch.finfo(min.dtype).max for _ in range(19)] + [(max.max() - min.max()).item()]

        while not self.should_stop_beta_gamma('BETA', set, min, max, last_gap):
            size_before = len(set)
            set, keep, prune_all = self.prune_beta_gamma(set, min, max)
            logger.debug(f'Min/max pruning {size_before - len(set)}')

            if prune_all:
                logger.warning(f'Pruning all in beta: {min}, {max}, last gap: {last_gap[-1]}')
                break

            lower = Affine(lower.A[keep], lower.b[keep], set.lower, set.upper)
            upper = Affine(upper.A[keep], upper.b[keep], set.lower, set.upper)

            set = self.split_beta(set, lower, upper)
            set = self.region_prune(set, self.dynamics.safe)
            lower, upper = bounds(self.beta_network, set, **kwargs)
            min, max = self.min_max_beta(lower, upper)

            last_gap.append((max.max() - min.max()).item())
            last_gap.pop(0)

        return max.max().clamp(min=0)

    def min_max_beta(self, lower, upper):
        min = lower.partition_min()
        max = upper.partition_max()

        return min.view(-1), max.view(-1)

    def split_beta(self, set, lower, upper):
        split_dim = ((lower.A.abs() + upper.A.abs())[:, 0] * set.width).argmax(dim=-1)
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
        last_gap = [torch.finfo(min.dtype).max for _ in range(9)] + [(max.max() - min.max()).item()]

        while not self.should_stop_beta_gamma('GAMMA', set, min, max, last_gap):
            set, _, prune_all = self.prune_beta_gamma(set, min, max)

            if prune_all:
                logger.warning(f'Pruning all in gamma: {min}, {max}, last gap: {last_gap[-1]}')
                break

            set = self.split(set, **kwargs)
            set = self.region_prune(set, self.dynamics.initial)

            min, max = self.min_max(set, **kwargs)

            last_gap.append((max.max() - min.max()).item())
            last_gap.pop(0)

        return max.max().clamp(min=0)

    def prune_beta_gamma(self, set, min, max):
        largest_lower_bound = min.max()

        prune = (max <= 0.0) | (max <= largest_lower_bound)
        keep = ~prune

        if torch.all(prune):
            return set, keep, True

        set = Partitions((set.lower[keep], set.upper[keep]))
        return set, keep, False

    def should_stop_beta_gamma(self, label, set, min, max, last_gap, max_set_size=None):
        gap = last_gap[-1]
        abs_max = max.max().item()

        max_set_size = max_set_size or self.max_set_size

        logger.debug(f'[{label}] Gap: {gap}, set size: {len(set)}, upper bound: {max.max().item()}')

        return len(set) > max_set_size or \
               abs_max <= 0.0 or \
               gap <= self.split_gap_stop_treshold or \
               gap >= torch.tensor(last_gap).max()

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
        return self.violation('UNSAFE', self.initial_partitioning.unsafe, 1, self.dynamics.unsafe, **kwargs)

    @torch.no_grad()
    def state_space_violation(self, **kwargs):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        if self.initial_partitioning.state_space is not None:
            return self.violation('STATE_SPACE', self.initial_partitioning.state_space, 0, self.dynamics.state_space, **kwargs)
        else:
            # Assume that dynamics ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
            return 0.0

    @torch.no_grad()
    def violation(self, label, set, lower_bound, contain_func, **kwargs):
        min, max = self.min_max(set, **kwargs)
        last_gap = [torch.finfo(min.dtype).max for _ in range(10)]

        while not self.should_stop_violation(label, set, min, max, lower_bound, last_gap):
            last_gap.append((max.min() - min.min()).item())
            last_gap.pop(0)

            set, prune_all = self.prune_violation(set, min, max, lower_bound)

            if prune_all:
                logger.warning(f'Pruning all in violation: {min}, {max}, last gap: {last_gap[-1]}')
                break

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

            lower, upper = bounds(self.barrier, set, method='crown_interval', **kwargs)
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
            return set, True

        return Partitions((set.lower[keep], set.upper[keep])), False

    def split(self, set, **kwargs):
        kwargs.pop('method', None)

        lower, upper = bounds(self.barrier, set, method='crown_linear', **kwargs)

        split_dim = ((lower.A.abs() + upper.A.abs())[:, 0] * set.width).argmax(dim=-1)
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

    def should_stop_violation(self, label, set, min, max, lower_bound, last_gap):
        gap = (max.min() - min.min()).item()
        abs_min = min.min().item()

        logger.debug(f'[{label}] Gap: {gap}, set size: {len(set)}, lower bound: {min.min().item()}')

        return len(set) > self.max_set_size or \
               abs_min >= lower_bound or \
               gap <= self.split_gap_stop_treshold or \
               gap >= torch.tensor(last_gap).max()

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

