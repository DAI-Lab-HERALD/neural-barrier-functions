import logging
import math

import torch
from bound_propagation import Clamp
from torch import nn

from .dynamics import AdditiveGaussianDynamics
from .bounds import bounds, Affine
from .partitioning import Partitions
from .networks import BetaNetwork, ExpectationBounds

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
    def __init__(self, barrier: nn.Module, dynamics: AdditiveGaussianDynamics, factory, initial_partitioning, beta_partitioning, horizon,
                 certification_threshold=1.0e-10, split_gap_stop_treshold=1e-6,
                 max_set_size=500000):
        super().__init__()

        assert isinstance(dynamics, AdditiveGaussianDynamics)

        self.factory = factory
        self.barrier = factory.build(barrier)
        self.nominal_dynamics = factory.build(dynamics.nominal_system)
        self.dynamics = dynamics

        # Assumptions:
        # 1. Initial set, unsafe set, and safe set are partitioned.
        # 2. State space is optionally partitioned. If not, it should end with ReLU.
        # 3. Partitions containing the boundary of the safe / unsafe set belong to both (to ensure correctness)
        # 4. Partitions are hyperrectangular and non-overlapping
        self.initial_partitioning = initial_partitioning
        self.beta_partitioning = beta_partitioning

        self.horizon = horizon
        self.certification_threshold = certification_threshold
        self.split_gap_stop_treshold = split_gap_stop_treshold
        self.max_set_size = max_set_size

    @torch.no_grad()
    def beta(self, **kwargs):
        assert self.initial_partitioning.safe is not None
        assert self.beta_partitioning.state_space is not None

        state_space_partitioning = self.beta_state_space_partitioning(**kwargs)
        # state_space_partitioning = self.beta_partitioning.state_space

        # _ = kwargs.pop('method')
        state_space_lower, state_space_upper = bounds(self.barrier, state_space_partitioning, method='crown_linear', **{key: value for key, value in kwargs.items() if key != 'method'})
        A_qi = state_space_lower.A, state_space_upper.A
        b_qi = state_space_lower.b, state_space_upper.b
        scale = self.dynamics.v[1].to(A_qi[0].device)
        q_bounds = state_space_partitioning.lower, state_space_partitioning.upper
        expectation_network = ExpectationBounds(self.barrier.module, self.dynamics.nominal_system, A_qi, b_qi, scale, q_bounds)
        expectation_network = self.factory.build(expectation_network)

        set = self.initial_partitioning.safe

        min, max = self.min_max_beta(expectation_network, set, **kwargs)
        last_gap = [torch.finfo(min.dtype).max for _ in range(9)] + [(max.max() - min.max()).item()]

        while not self.should_stop_beta_gamma('BETA', set, min, max, last_gap):
            size_before = len(set)
            set, prune_all = self.prune_beta_gamma(set, min, max)
            logger.debug(f'Min/max pruning {size_before - len(set)}')

            if prune_all:
                logger.warning(f'Pruning all in beta: {min}, {max}, last gap: {last_gap[-1]}')
                break

            batch_size = 100

            new_sets = []
            new_mins = []
            new_maxs = []

            while len(set) > 0:
                new_set = self.split_beta(set[:batch_size], **kwargs)
                new_set = self.region_prune(new_set, self.dynamics.safe)
                new_sets.append(new_set)

                new_min, new_max = self.min_max_beta(expectation_network, new_set, **kwargs)
                new_mins.append(new_min)
                new_maxs.append(new_max)

                set = set[batch_size:]

            set = Partitions((torch.cat([set.lower for set in new_sets]), torch.cat([set.upper for set in new_sets])))
            min, max = torch.cat(new_mins), torch.cat(new_maxs)

            last_gap.append((max.max() - min.max()).item())
            last_gap.pop(0)

        return max.max().clamp(min=0)

    def min_max_beta(self, expectation_network, set, **kwargs):
        lower, upper = bounds(expectation_network, set, **kwargs)
        min = lower.partition_min()
        max = upper.partition_max()

        return min.view(-1), max.view(-1)

        # safe_lower, safe_upper = bounds(self.barrier, set, method='crown_linear', **kwargs)
        # dynamics_lower, dynamics_upper = bounds(self.nominal_dynamics, set, method='crown_linear', **kwargs)
        #
        # P_v_in_q = self.P_v_in_q(state_space_set, dynamics_lower, dynamics_upper)
        #
        # min = self.beta_min_max_oneside(state_space_set, P_v_in_q[0], state_space_lower, dynamics_lower, dynamics_upper, safe_upper)
        # max = self.beta_min_max_oneside(state_space_set, P_v_in_q[1], state_space_upper, dynamics_lower, dynamics_upper, safe_lower, min=False)
        #
        # return min, max

    def beta_min_max_oneside(self, state_space_set, P_v_in_q, state_space_bounds, dynamics_lower, dynamics_upper, safe, min=True):
        nonzero_prob = P_v_in_q.squeeze(-1).nonzero(as_tuple=True)
        PA_sum = torch.zeros((P_v_in_q.size(0), 1, state_space_bounds.A.size(-1)), device=P_v_in_q.device)
        index = torch.arange(P_v_in_q.size(0), device=P_v_in_q.device).view(-1, 1).expand(-1, state_space_bounds.A.size(0))
        index = index[nonzero_prob]

        A = state_space_bounds.A.unsqueeze(0).expand(P_v_in_q.size(0), -1, -1, -1)
        PA = P_v_in_q[nonzero_prob].unsqueeze(-1) * A[nonzero_prob]

        PA_sum.scatter_add_(0, index.view(-1, 1, 1).expand(-1, 1, A.size(-1)), PA)

        # PA = P_v_in_q.unsqueeze(-2) * state_space_bounds.A
        # PA_sum = PA_top.sum(dim=1)

        first_term_A_lower = PA_sum.matmul(dynamics_lower.A)
        first_term_b_lower = PA_sum.matmul(dynamics_lower.b.unsqueeze(-1)).squeeze(-1)

        first_term_A_upper = PA_sum.matmul(dynamics_upper.A)
        first_term_b_upper = PA_sum.matmul(dynamics_upper.b.unsqueeze(-1)).squeeze(-1)

        second_term_b = (P_v_in_q * state_space_bounds.b).sum(dim=1)

        third_term = self.barrier_noise(state_space_set, state_space_bounds, dynamics_lower, dynamics_upper)
        if min:
            third_term_b = third_term[0]
        else:
            third_term_b = third_term[1]

        BFx_for_F_lower = \
            Affine(first_term_A_lower - safe.A, first_term_b_lower + second_term_b + third_term_b - safe.b,
                      safe.lower, safe.upper)

        BFx_for_F_upper = \
            Affine(first_term_A_upper - safe.A, first_term_b_upper + second_term_b + third_term_b - safe.b,
                      safe.lower, safe.upper)

        if min:
            return torch.min(BFx_for_F_lower.partition_min(), BFx_for_F_upper.partition_min()).view(-1)
        else:
            return torch.max(BFx_for_F_lower.partition_max(), BFx_for_F_upper.partition_max()).view(-1)

    def v_bounds(self, set, dynamics_lower_interval, dynamics_upper_interval):
        dynamics_lower_interval = dynamics_lower_interval.unsqueeze(1)
        dynamics_upper_interval = dynamics_upper_interval.unsqueeze(1)

        v_bounds_lower = set.lower - dynamics_lower_interval, set.upper - dynamics_upper_interval
        v_bounds_upper = set.lower - dynamics_upper_interval, set.upper - dynamics_lower_interval

        return v_bounds_lower, v_bounds_upper

    def barrier_noise(self, set, state_space_bounds, dynamics_lower, dynamics_upper):
        _, scale = self.dynamics.v
        scale = scale.to(dynamics_lower.A.device)
        nonzero_scale = (scale > 0.0)

        q_bounds = set.lower[..., nonzero_scale], set.upper[..., nonzero_scale]
        A = state_space_bounds.A[..., nonzero_scale]

        scale = scale[nonzero_scale]
        prefix_factor = scale / math.sqrt(2 * math.pi)

        dynamics_lower_interval = dynamics_lower.partition_min()[..., nonzero_scale].unsqueeze(1) / (2 * scale)
        dynamics_upper_interval = dynamics_upper.partition_max()[..., nonzero_scale].unsqueeze(1) / (2 * scale)
        q_bounds = q_bounds[0] / (2 * scale), q_bounds[1] / (2 * scale)

        v_lower_minus_dynamics_bounds = q_bounds[0] - dynamics_upper_interval, q_bounds[0] - dynamics_lower_interval
        v_upper_minus_dynamics_bounds = q_bounds[1] - dynamics_upper_interval, q_bounds[1] - dynamics_lower_interval

        v_lower_minus_dynamics_bounds_squared = v_lower_minus_dynamics_bounds[0] ** 2, v_lower_minus_dynamics_bounds[1] ** 2
        v_upper_minus_dynamics_bounds_squared = v_upper_minus_dynamics_bounds[0] ** 2, v_upper_minus_dynamics_bounds[1] ** 2

        v_lower_minus_dynamics_bounds_cross_zero = (v_lower_minus_dynamics_bounds[0] < 0.0) & (0.0 < v_lower_minus_dynamics_bounds[1])
        v_lower_squared_bounds = (~v_lower_minus_dynamics_bounds_cross_zero) * \
                                 torch.min(*v_lower_minus_dynamics_bounds_squared), \
                                 torch.max(*v_lower_minus_dynamics_bounds_squared)

        v_upper_minus_dynamics_bounds_cross_zero = (v_upper_minus_dynamics_bounds[0] < 0.0) & (0.0 < v_upper_minus_dynamics_bounds[1])
        v_upper_squared_bounds = (~v_upper_minus_dynamics_bounds_cross_zero) * \
                                 torch.min(*v_upper_minus_dynamics_bounds_squared), \
                                 torch.max(*v_upper_minus_dynamics_bounds_squared)

        first_exponent_bounds = -2 * v_lower_squared_bounds[1], -2 * v_lower_squared_bounds[0]
        second_exponent_bounds = -2 * v_upper_squared_bounds[1], -2 * v_upper_squared_bounds[0]

        first_exponential_bounds = first_exponent_bounds[0].exp(), first_exponent_bounds[1].exp()
        second_exponential_bounds = second_exponent_bounds[0].exp(), second_exponent_bounds[1].exp()

        # Notice that we flip the exponential bounds because the prefix contains a negative factor
        combined_bounds = prefix_factor * (first_exponential_bounds[0] - second_exponential_bounds[1]),\
                          prefix_factor * (first_exponential_bounds[1] - second_exponential_bounds[0])

        mid = (combined_bounds[0] + combined_bounds[1]) / 2
        diff = (combined_bounds[1] - combined_bounds[0]) / 2

        lower = A.matmul(mid.unsqueeze(-1)) - A.abs().matmul(diff.unsqueeze(-1))
        upper = A.matmul(mid.unsqueeze(-1)) + A.abs().matmul(diff.unsqueeze(-1))

        return lower.squeeze(-1).sum(dim=1), upper.squeeze(-1).sum(dim=1)

    def P_v_in_q(self, set, dynamics_lower, dynamics_upper):
        loc, scale = self.dynamics.v
        loc, scale = loc.to(dynamics_lower.A.device), scale.to(dynamics_lower.A.device)
        nonzero_scale = (scale > 0.0)
        zero_scale = (scale == 0.0)

        q_bounds = set.lower, set.upper

        zero_scale_loc = loc[zero_scale]
        loc, scale = loc[nonzero_scale], scale[nonzero_scale]

        dynamics_lower_interval = dynamics_lower.partition_min()
        dynamics_upper_interval = dynamics_upper.partition_max()

        v_bounds_lower, v_bounds_upper = self.v_bounds(set, dynamics_lower_interval, dynamics_upper_interval)

        dynamics_lower_interval = dynamics_lower_interval[..., nonzero_scale].unsqueeze(1)
        dynamics_upper_interval = dynamics_upper_interval[..., nonzero_scale].unsqueeze(1)

        q_lower = (q_bounds[0][..., nonzero_scale] - dynamics_upper_interval - loc) / (scale * math.sqrt(2.0)), \
                  (q_bounds[0][..., nonzero_scale] - dynamics_lower_interval - loc) / (scale * math.sqrt(2.0))

        q_upper = (q_bounds[1][..., nonzero_scale] - dynamics_upper_interval - loc) / (scale * math.sqrt(2.0)), \
                  (q_bounds[1][..., nonzero_scale] - dynamics_lower_interval - loc) / (scale * math.sqrt(2.0))

        # We can do this because erf is monotonously increasing function
        q_lower_erf = torch.erf(q_lower[0]), torch.erf(q_lower[1])
        q_upper_erf = torch.erf(q_upper[0]), torch.erf(q_upper[1])

        lower = (q_upper_erf[0] - q_lower_erf[1]) / 2.0
        upper = (q_upper_erf[1] - q_lower_erf[0]) / 2.0

        lower = lower.prod(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
        upper = upper.prod(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)

        zero = torch.any((v_bounds_lower[0][..., zero_scale] > zero_scale_loc) | (v_bounds_lower[1][..., zero_scale] < zero_scale_loc) | (v_bounds_lower[0][..., zero_scale] > v_bounds_lower[1][..., zero_scale]), dim=-1)
        lower[zero] = 0.0

        zero = torch.any((v_bounds_upper[0][..., zero_scale] > zero_scale_loc) | (v_bounds_upper[1][..., zero_scale] < zero_scale_loc) | (v_bounds_upper[0][..., zero_scale] > v_bounds_upper[1][..., zero_scale]), dim=-1)
        upper[zero] = 0.0

        return lower, upper

    def split_beta(self, set, **kwargs):
        kwargs.pop('method', None)

        lower_dynamics, upper_dynamics = bounds(self.nominal_dynamics, set, method='crown_linear', **kwargs)
        lower_barrier, upper_barrier = bounds(self.barrier, set, method='crown_linear', **kwargs)

        split_dim = ((lower_dynamics.A.sum(dim=-1, keepdim=True).abs() + upper_dynamics.A.sum(dim=-1, keepdim=True).abs() + lower_barrier.A.abs() + upper_barrier.A.abs())[:, 0] * set.width).argmax(dim=-1)
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

    def beta_state_space_partitioning(self, **kwargs):
        assert self.initial_partitioning.state_space is not None
        set = self.initial_partitioning.state_space

        min, max = self.min_max(set, **kwargs)
        lower, upper = bounds(self.barrier, set, method='crown_linear', **{key: value for key, value in kwargs.items() if key != 'method'})
        last_gap = [torch.finfo(min.dtype).max for _ in range(199)] + [(upper - lower).partition_max().max().item()]

        while not self.should_stop_beta_gamma('BETA_STATE_SPACE', set, min, max, last_gap, max_set_size=10000):
            batch_size = 100
            k = torch.min(torch.tensor([batch_size, len(set)])).item()

            split_indices = self.split_indices(set, k, min, max, lower, upper)
            other_indices = torch.full_like(min, True, dtype=torch.bool)
            other_indices[split_indices] = False

            split_set = set[split_indices]

            set = set[other_indices]
            min, max = min[other_indices], max[other_indices]
            lower = (lower.A[other_indices], lower.b[other_indices], lower.lower[other_indices], lower.upper[other_indices])
            upper = (upper.A[other_indices], upper.b[other_indices], upper.lower[other_indices], upper.upper[other_indices])

            split_set = self.split(split_set, **kwargs)
            split_set = self.region_prune(split_set, self.dynamics.state_space)

            split_min, split_max = self.min_max(split_set, **kwargs)
            min, max = torch.cat([min, split_min]), torch.cat([max, split_max])

            set = Partitions((torch.cat([set.lower, split_set.lower]), torch.cat([set.upper, split_set.upper])))

            split_lower, split_upper = bounds(self.barrier, split_set, method='crown_linear', **{key: value for key, value in kwargs.items() if key != 'method'})
            lower = Affine(torch.cat([lower[0], split_lower.A]), torch.cat([lower[1], split_lower.b]), torch.cat([lower[2], split_lower.lower]), torch.cat([lower[3], split_lower.upper]))
            upper = Affine(torch.cat([upper[0], split_upper.A]), torch.cat([upper[1], split_upper.b]), torch.cat([upper[2], split_upper.lower]), torch.cat([upper[3], split_upper.upper]))

            last_gap.append((upper - lower).partition_max().max().item())
            last_gap.pop(0)

        return set

    def split_indices(self, set, k, min, max, lower, upper):
        #largest_average_slope = ((lower.A.abs() + upper.A.abs())[:, 0] * set.width).sum(dim=-1).topk(k)
        largest_linear_gap = ((upper - lower).partition_max()[:, 0] * set.volume).topk(k)
        # largest_interval_gap = (max - min).topk(k)
        return largest_linear_gap.indices

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
            set, prune_all = self.prune_beta_gamma(set, min, max)

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
            return set, True

        return Partitions((set.lower[keep], set.upper[keep])), False

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

