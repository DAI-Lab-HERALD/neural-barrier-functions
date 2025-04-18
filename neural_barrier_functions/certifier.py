import logging
import math

import torch
from bound_propagation import Clamp, IntervalBounds, LinearBounds, HyperRectangle
from torch import nn

from .dynamics import AdditiveGaussianDynamics
from .bounds import bounds, Affine, NBFBoundModelFactory
from .partitioning import Partitions
from .networks import BetaNetwork, AdditiveGaussianBetaNetwork

logger = logging.getLogger(__name__)


class SplittingNeuralSBFCertifier(nn.Module):
    def __init__(self, barrier, dynamics, factory, partitioning, horizon,
                 certification_threshold=1.0e-10, split_gap_stop_treshold=1e-6,
                 max_set_size=10000):
        super().__init__()

        # barrier = nn.Sequential(barrier, Clamp(max=1.0 + 1e-6))

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

            keep, prune_all = self.prune_beta_gamma(min, max)

            if prune_all:
                logger.warning(f'Pruning all in beta: {min}, {max}, last gap: {last_gap[-1]}')
                break

            set = set[keep]

            set = self.split_beta(set, **kwargs)
            set = self.region_prune(set, self.dynamics.safe)

            min, max = self.min_max_beta(set, **kwargs)

        return max.max().clamp(min=0.0)

    def min_max_beta(self, set, **kwargs):
        lower, upper = bounds(self.beta_network, set, **kwargs)
        min = lower.partition_min()
        max = upper.partition_max()

        return min.view(-1), max.view(-1)

    def split_beta(self, set, **kwargs):
        if kwargs.get('method') not in ['crown_linear', 'crown_ibp_linear']:
            kwargs['method'] = 'crown_ibp_linear'

        lower, upper = bounds(self.beta_network, set, **kwargs)

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

        min, max = self.min_max_gamma(set, **kwargs)
        last_gap = [torch.finfo(min.dtype).max for _ in range(10)]

        while not self.should_stop_beta_gamma('GAMMA', set, min, max, last_gap):
            last_gap.append((max.max() - min.max()).item())
            last_gap.pop(0)

            keep, prune_all = self.prune_beta_gamma(min, max)

            if prune_all:
                logger.warning(f'Pruning all in gamma: {min}, {max}, last gap: {last_gap[-1]}')
                break

            set = set[keep]

            set = self.split(set, **kwargs)
            set = self.region_prune(set, self.dynamics.initial)

            min, max = self.min_max_gamma(set, **kwargs)

        return max.max().clamp(min=0.0)

    def min_max_gamma(self, set, **kwargs):
        lower, upper = bounds(self.barrier, set, **kwargs)
        min = lower.partition_min()
        max = upper.partition_max()

        return min.view(-1), max.view(-1)

    def prune_beta_gamma(self, min, max):
        largest_lower_bound = min.max()

        prune = (max <= 0.0) | (max <= largest_lower_bound)
        keep = ~prune

        return keep, torch.all(prune)

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
        ss_violation, ss_ce = self.state_space_violation(**kwargs)
        unsafe_violation, unsafe_ce = self.unsafe_violation(**kwargs)
        violation = ss_violation + unsafe_violation
        return violation, (ss_ce, unsafe_ce)

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
            return 0.0, None

    @torch.no_grad()
    def violation(self, label, set, lower_bound, contain_func, **kwargs):
        min, max = self.min_max(set, **kwargs)
        last_gap = [torch.finfo(min.dtype).max for _ in range(10)]

        while not self.should_stop_violation(label, set, min, max, lower_bound, last_gap):
            keep, prune_all = self.prune_violation(min, max, lower_bound)

            if prune_all:
                logger.warning(f'Pruning all in violation: {min}, {max}, last gap: {last_gap[-1]}')
                break

            set = set[keep]

            set = self.split(set, **kwargs)
            set = self.region_prune(set, contain_func)

            min, max = self.min_max(set)

            last_gap.append((max.min() - min.min()).item())
            last_gap.pop(0)

        return (lower_bound - min.min()).clamp(min=0.0), set[min.argmin()].center

    def min_max(self, set, **kwargs):
        lower, upper = bounds(self.barrier, set, **kwargs)
        min = lower.partition_min()
        max = upper.partition_max()

        return min.view(-1), max.view(-1)

    def prune_violation(self, min, max, lower_bound):
        least_upper_bound = max.min()

        prune = (min >= lower_bound) | (min >= least_upper_bound)
        keep = ~prune

        return keep, torch.all(prune)

    def split(self, set, **kwargs):
        if kwargs.get('method') not in ['crown_linear', 'crown_ibp_linear']:
            kwargs['method'] = 'crown_ibp_linear'

        lower, upper = bounds(self.barrier, set, **kwargs)

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
        gamma = self.gamma(**kwargs)
        beta = self.beta(**kwargs)
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
        violation, ce = self.barrier_violation(**kwargs)
        violation = violation.item()
        certified = violation <= self.certification_threshold

        return certified, violation, ce


class AdditiveGaussianSplittingNeuralSBFCertifier(nn.Module):
    def __init__(self, barrier: nn.Module, dynamics: AdditiveGaussianDynamics, factory, initial_partitioning, horizon,
                 certification_threshold=1.0e-6, split_gap_stop_treshold=1e-6, max_set_size=200000,
                 noise_partitions=200, sigma_cutoff=3.0, device=None):
        super().__init__()

        assert isinstance(dynamics, AdditiveGaussianDynamics)

        self.barrier = factory.build(barrier)

        loc, scale = dynamics.v
        loc, scale = loc.to(device), scale.to(device)
        factory.kwargs['slices'] = [noise_partitions for _ in range(initial_partitioning.state_space.size(-1))]
        factory.kwargs['sigma_cutoff'] = sigma_cutoff
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

        lower, upper = bounds(self.beta_network, set, **kwargs)
        min, max = self.min_max_beta(set, lower, upper)
        last_gap = [torch.finfo(min.dtype).max for _ in range(499)] + [(max.max() - min.max()).item()]
        k = 0

        while not self.should_stop_beta_gamma('BETA', set, min, max, last_gap):
            partition_gap = max - min
            logger.debug(f'Partition gap (min: {partition_gap.min().item()}/max: {partition_gap.max().item()})')

            keep, prune_all = self.prune_beta_gamma(min, max)

            if prune_all:
                logger.warning(f'Pruning all in beta: {min}, {max}, last gap: {last_gap[-1]}')
                break

            size_before = len(set)

            set = set[keep]
            lower = lower[keep]
            upper = upper[keep]
            logger.debug(f'Min/max pruning {size_before - len(set)}')

            split_index, other = self.pick_for_splitting(min[keep], max[keep], **kwargs)

            split_set = set[split_index]
            split_lower = lower[split_index]
            split_upper = upper[split_index]

            set = set[other]
            lower = lower[other]
            upper = upper[other]

            split_set = self.split(split_set, split_lower, split_upper, k)
            split_set = self.region_prune(split_set, self.dynamics.safe)
            split_lower, split_upper = bounds(self.beta_network, split_set, **kwargs)

            set = Partitions((torch.cat([set.lower, split_set.lower]), torch.cat([set.upper, split_set.upper])))
            lower_A = 0.0 if isinstance(lower.A, float) else torch.cat([lower.A, split_lower.A])
            lower = Affine(lower_A, torch.cat([lower.b, split_lower.b]), set)
            upper_A = 0.0 if isinstance(upper.A, float) else torch.cat([upper.A, split_upper.A])
            upper = Affine(upper_A, torch.cat([upper.b, split_upper.b]), set)

            min, max = self.min_max_beta(set, lower, upper)

            last_gap.append((max.max() - min.max()).item())
            last_gap.pop(0)
            k += 1

        return max.max().clamp(min=0)

    def pick_for_splitting(self, min_, max_, batch_size=1, **kwargs):
        split_indices = max_.topk(min(batch_size, max_.size(0))).indices

        other_indices = torch.full((max_.size(0),), True, dtype=torch.bool, device=max_.device)
        other_indices[split_indices] = False

        return split_indices, other_indices

    def min_max_beta(self, set, lower, upper):
        minmin = lower.partition_min().view(-1)
        minmax = lower.partition_max().view(-1)
        min = torch.where(self.dynamics.unsafe(set.center, set.width / 2), minmin, minmax)
        max = upper.partition_max()

        return min, max.view(-1)

    def min_max(self, lower, upper):
        min = lower.partition_min()
        max = upper.partition_max()

        return min.view(-1), max.view(-1)

    def split(self, set, lower, upper, k):
        if isinstance(lower.A, float):
            split_dim = (..., k % set.size(-1))
        else:
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

        lower, upper = bounds(self.barrier, set, **kwargs)
        min, max = self.min_max(lower, upper)
        last_gap = [torch.finfo(min.dtype).max for _ in range(499)] + [(max.max() - min.max()).item()]
        k = 0

        while not self.should_stop_beta_gamma('GAMMA', set, min, max, last_gap):
            keep, prune_all = self.prune_beta_gamma(min, max)

            if prune_all:
                logger.warning(f'Pruning all in gamma: {min}, {max}, last gap: {last_gap[-1]}')
                break

            set = set[keep]
            lower = lower[keep]
            upper = upper[keep]

            split_index, other = self.pick_for_splitting(min[keep], max[keep], **kwargs)

            split_set = set[split_index]
            split_lower = lower[split_index]
            split_upper = upper[split_index]

            set = set[other]
            lower = lower[other]
            upper = upper[other]

            split_set = self.split(split_set, split_lower, split_upper, k)
            split_set = self.region_prune(split_set, self.dynamics.initial)
            split_lower, split_upper = bounds(self.barrier, split_set, **kwargs)

            set = Partitions((torch.cat([set.lower, split_set.lower]), torch.cat([set.upper, split_set.upper])))
            lower_A = 0.0 if isinstance(lower.A, float) else torch.cat([lower.A, split_lower.A])
            lower = Affine(lower_A, torch.cat([lower.b, split_lower.b]), set)
            upper_A = 0.0 if isinstance(upper.A, float) else torch.cat([upper.A, split_upper.A])
            upper = Affine(upper_A, torch.cat([upper.b, split_upper.b]), set)

            min, max = self.min_max(lower, upper)

            last_gap.append((max.max() - min.max()).item())
            last_gap.pop(0)
            k += 1

        return max.max().clamp(min=0)

    def prune_beta_gamma(self, min, max):
        largest_lower_bound = min.max()

        prune = (max <= 0.0) | (max <= largest_lower_bound)
        keep = ~prune

        return keep, torch.all(prune)

    def should_stop_beta_gamma(self, label, set, min, max, last_gap, max_set_size=None):
        gap = last_gap[-1]
        abs_max = max.max().item()

        max_set_size = max_set_size or self.max_set_size

        logger.debug(f'[{label}] Gap: {gap}, set size: {len(set)}, upper bound: {max.max().item()}')

        return len(set) > max_set_size or \
               abs_max <= 0.0 or \
               gap <= self.split_gap_stop_treshold or \
               torch.tensor(last_gap[-10:]).min() >= torch.tensor(last_gap).max()

    @torch.no_grad()
    def barrier_violation(self, **kwargs):
        ss_violation, ss_ce = self.state_space_violation(**kwargs)
        unsafe_violation, unsafe_ce = self.unsafe_violation(**kwargs)
        violation = ss_violation + unsafe_violation
        return violation, (ss_ce, unsafe_ce)

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
            return 0.0, None

    @torch.no_grad()
    def violation(self, label, set, lower_bound, contain_func, **kwargs):
        lower, upper = bounds(self.barrier, set, **kwargs)
        min, max = self.min_max(lower, upper)
        last_gap = [torch.finfo(min.dtype).max for _ in range(499)]
        k = 0

        while not self.should_stop_violation(label, set, min, max, lower_bound, last_gap):
            keep, prune_all = self.prune_violation(min, max, lower_bound)

            if prune_all:
                logger.warning(f'Pruning all in violation: {min}, {max}, last gap: {last_gap[-1]}')
                break

            set = set[keep]
            lower = lower[keep]
            upper = upper[keep]

            split_index, other = self.pick_for_splitting_violation(min[keep], max[keep], **kwargs)

            split_set = set[split_index]
            split_lower = lower[split_index]
            split_upper = upper[split_index]

            set = set[other]
            lower = lower[other]
            upper = upper[other]

            split_set = self.split(split_set, split_lower, split_upper, k)
            split_set = self.region_prune(split_set, contain_func)
            split_lower, split_upper = bounds(self.barrier, split_set, **kwargs)

            set = Partitions((torch.cat([set.lower, split_set.lower]), torch.cat([set.upper, split_set.upper])))
            lower_A = 0.0 if isinstance(lower.A, float) else torch.cat([lower.A, split_lower.A])
            lower = Affine(lower_A, torch.cat([lower.b, split_lower.b]), set)
            upper_A = 0.0 if isinstance(upper.A, float) else torch.cat([upper.A, split_upper.A])
            upper = Affine(upper_A, torch.cat([upper.b, split_upper.b]), set)

            min, max = self.min_max(lower, upper)

            last_gap.append((max.min() - min.min()).item())
            last_gap.pop(0)
            k += 1

        return (lower_bound - min.min()).clamp(min=0.0), set[min.argmin()].center

    def pick_for_splitting_violation(self, min_, max_, batch_size=1, **kwargs):
        split_indices = min_.topk(min(batch_size, min_.size(0)), largest=False).indices

        other_indices = torch.full((max_.size(0),), True, dtype=torch.bool, device=max_.device)
        other_indices[split_indices] = False

        return split_indices, other_indices

    def prune_violation(self, min, max, lower_bound):
        least_upper_bound = max.min()

        prune = (min >= lower_bound) | (min >= least_upper_bound)
        keep = ~prune

        return keep, torch.all(prune)

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
        gamma = self.gamma(**kwargs)
        beta = self.beta(**kwargs)
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
        violation, ce = self.barrier_violation(**kwargs)
        violation = violation.item()
        certified = violation <= self.certification_threshold

        return certified, violation, ce

