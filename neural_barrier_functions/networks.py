import copy
import math

import torch
from bound_propagation import Residual, BoundModule, LinearBounds, IntervalBounds, HyperRectangle, Sub
from scipy import stats
from torch import nn
from torch.distributions import Normal


class Mean(nn.Module):
    def __init__(self, subnetwork):
        super().__init__()

        self.subnetwork = subnetwork

    def forward(self, x):
        return self.subnetwork(x).mean(dim=0)


class BoundMean(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.subnetwork = factory.build(module.subnetwork)

    @property
    def need_relaxation(self):
        return self.subnetwork.need_relaxation

    def clear_relaxation(self):
        self.subnetwork.clear_relaxation()

    def backward_relaxation(self, region):
        return self.subnetwork.backward_relaxation(region)

    def crown_backward(self, linear_bounds, optimize):
        subnetwork_bounds = self.subnetwork.crown_backward(linear_bounds, optimize)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (subnetwork_bounds.lower[0].mean(dim=0), subnetwork_bounds.lower[1].mean(dim=0))

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (subnetwork_bounds.upper[0].mean(dim=0), subnetwork_bounds.upper[1].mean(dim=0))

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        subnetwork_bounds = self.subnetwork.ibp_forward(bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)
        return IntervalBounds(bounds.region, subnetwork_bounds.lower.mean(dim=0), subnetwork_bounds.upper.mean(dim=0))

    def propagate_size(self, in_size):
        return self.subnetwork.propagate_size(in_size)


class BetaNetwork(Sub):
    def __init__(self, dynamics, barrier):
        super().__init__(Mean(nn.Sequential(dynamics, barrier)), barrier)

    def forward(self, x):
        return self.dynamics_barrier(x) - self.barrier(x)


class Sum(nn.Module):
    def forward(self, x):
        return x.sum(dim=0)


class BoundSum(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (linear_bounds.lower[0].sum(dim=0), linear_bounds.lower[1].sum(dim=0))

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (linear_bounds.upper[0].sum(dim=0), linear_bounds.upper[1].sum(dim=0))

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        return IntervalBounds(bounds.region, bounds.lower.sum(dim=0), bounds.upper.sum(dim=0))

    def propagate_size(self, in_size):
        return in_size


class VRegionMixin:
    def v_regions(self, device):
        loc, scale = self.module.loc, self.module.scale

        v_min = loc - self.sigma_cutoff * scale
        v_max = loc + self.sigma_cutoff * scale

        centers = []
        half_widths = []
        probs = []
        cond_exps = []

        for i, num_slices in enumerate(self.slices):
            if scale[i].item() == 0.0:
                center = torch.full((1,), loc[i])
                half_width = torch.zeros((1,))
                prob = torch.ones((1,))
                cond_exp = center
            else:
                v_space, prob, cond_exp = gaussian_partitioning(v_min[i], v_max[i], num_slices, loc[i], scale[i])
                center = (v_space[:-1] + v_space[1:]) / 2
                half_width = (v_space[1:] - v_space[:-1]) / 2

            centers.append(center)
            half_widths.append(half_width)
            probs.append(prob)
            cond_exps.append(cond_exp)

        centers = torch.cartesian_prod(*centers).to(device)
        half_widths = torch.cartesian_prod(*half_widths).to(device)
        lower, upper = centers - half_widths, centers + half_widths

        probs = torch.cartesian_prod(*probs).to(device)
        probs = probs.prod(-1)

        cond_exps = torch.cartesian_prod(*cond_exps).to(device)
        part_exps = cond_exps * probs.unsqueeze(-1)

        return HyperRectangle(lower, upper), probs, part_exps


def gaussian_partitioning(start, stop, slices, loc, scale):
    dist = Normal(loc, scale)
    steps = torch.linspace(0.0, 1.0, slices + 1)
    out = dist.icdf(steps * (dist.cdf(stop) - dist.cdf(start)) + dist.cdf(start))
    out[0], out[-1] = start, stop

    prob = dist.cdf(out[1:]) - dist.cdf(out[:-1])

    transformed_points = (out - loc) / scale
    transformed_a, transformed_b = transformed_points[:-1], transformed_points[1:]
    cond_exps = [stats.truncnorm.mean(a, b, loc=loc, scale=scale) for (a, b) in zip(transformed_a, transformed_b)]
    cond_exp = torch.as_tensor(cond_exps, dtype=out.dtype)

    return out, prob, cond_exp


class AdditiveGaussianExpectation(nn.Module):
    def __init__(self, barrier, nominal_dynamics, loc, scale):
        super().__init__()

        self.sum_module = Sum()
        self.barrier = barrier
        self.nominal_dynamics = nominal_dynamics
        self.loc, self.scale = loc, scale

    def forward(self, x):
        # NOTE: The expectation cannot be computed for a specific x.
        # We only have this module because we can _bound_ the expectation,
        # and thus need the bound module below.
        raise NotImplementedError()


class BoundAdditiveGaussianExpectation(BoundModule, VRegionMixin):
    def __init__(self, module, factory, sigma_cutoff=7.2, slices=None, barrier_clamp=1 + 1e-6, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_sum = factory.build(module.sum_module)
        self.bound_barrier = factory.build(module.barrier)
        self.bound_nominal_dynamics = factory.build(module.nominal_dynamics)

        assert sigma_cutoff > 0.0
        self.sigma_cutoff = sigma_cutoff

        assert isinstance(slices, list)
        self.slices = slices

        assert barrier_clamp >= 1.0
        self.barrier_clamp = barrier_clamp

    @property
    def loc(self):
        return self.module.loc

    @property
    def scale(self):
        return self.module.scale

    @property
    def need_relaxation(self):
        return self.bound_barrier.need_relaxation or self.bound_nominal_dynamics.need_relaxation

    def clear_relaxation(self):
        self.bound_barrier.clear_relaxation()
        self.bound_nominal_dynamics.clear_relaxation()

    def backward_relaxation(self, region):
        if self.bound_nominal_dynamics.need_relaxation:
            return self.bound_nominal_dynamics.backward_relaxation(region)

        linear_bounds, module, *extra = self.bound_barrier.backward_relaxation(region)
        dynamics_linear_bounds = self.bound_nominal_dynamics.crown_backward(linear_bounds, False)

        v_regions, _, _ = self.v_regions(region.lower.device)

        num_input_regions = len(region)
        num_noise_regions = len(v_regions)

        combined_region = HyperRectangle(
            torch.cat((
                region.lower.unsqueeze(0).expand(num_noise_regions, -1, -1),
                v_regions.lower.unsqueeze(1).expand(-1, num_input_regions, -1)
            ), dim=-1),
            torch.cat((
                region.upper.unsqueeze(0).expand(num_noise_regions, -1, -1),
                v_regions.upper.unsqueeze(1).expand(-1, num_input_regions, -1)
            ), dim=-1)
        )

        combined_linear_bounds = LinearBounds(
            combined_region,
            (torch.cat((dynamics_linear_bounds.lower[0], linear_bounds.lower[0]), dim=-1), dynamics_linear_bounds.lower[1]),
            (torch.cat((dynamics_linear_bounds.upper[0], linear_bounds.upper[0]), dim=-1), dynamics_linear_bounds.upper[1]),
        )

        return combined_linear_bounds, module, *extra

    def crown_backward(self, linear_bounds, optimize):
        linear_bounds = self.bound_barrier.crown_backward(linear_bounds, optimize)
        dynamics_linear_bounds = self.bound_nominal_dynamics.crown_backward(linear_bounds, optimize)

        _ , probs, part_exps = self.v_regions(linear_bounds.region.lower.device)

        if linear_bounds.lower is not None:
            lower_Av = linear_bounds.lower[0]
            lower_Ax = dynamics_linear_bounds.lower[0]
            lower_b = dynamics_linear_bounds.lower[1]

            lower_A = lower_Ax * probs.view(-1, 1, 1, 1)
            lower_b_noise = lower_Av.matmul(part_exps.view(part_exps.size(0), 1, -1, 1)).squeeze(-1)
            lower_b = lower_b * probs.view(-1, 1, 1) + lower_b_noise

            lower = (lower_A, lower_b)
        else:
            lower = None

        if linear_bounds.upper is not None:
            upper_Av = linear_bounds.upper[0]
            upper_Ax = dynamics_linear_bounds.upper[0]
            upper_b = dynamics_linear_bounds.upper[1]

            upper_A = upper_Ax * probs.view(-1, 1, 1, 1)
            upper_b_noise = upper_Av.matmul(part_exps.view(part_exps.size(0), 1, -1, 1)).squeeze(-1)
            upper_b = upper_b * probs.view(-1, 1, 1) + upper_b_noise

            upper = (upper_A, upper_b)
        else:
            upper = None

        linear_bounds = LinearBounds(linear_bounds.region, lower, upper)
        linear_bounds = self.bound_sum.crown_backward(linear_bounds, optimize)

        # Do not add to lower bound (B = 0 is a lower bound for the noise regions outside the cutoff).
        if linear_bounds.upper is not None:
            prob_outside = self.prob_outside()
            upper = (linear_bounds.upper[0], linear_bounds.upper[1] + self.barrier_clamp * prob_outside)

            linear_bounds = LinearBounds(linear_bounds.region, linear_bounds.lower, upper)

        return linear_bounds

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        v_regions, probs, _ = self.v_regions(bounds.lower.device)

        nominal_bounds = self.bound_nominal_dynamics.ibp_forward(bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)
        dynamics_bounds = IntervalBounds(
            bounds.region,
            nominal_bounds.lower + v_regions.lower.unsqueeze(-2),
            nominal_bounds.upper + v_regions.upper.unsqueeze(-2),
        )

        barrier_bounds = self.bound_barrier.ibp_forward(dynamics_bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)

        weighted_bounds = IntervalBounds(
            bounds.region,
            barrier_bounds.lower * probs.view(-1, 1, 1),
            barrier_bounds.upper * probs.view(-1, 1, 1)
        )
        total_bounds = self.bound_sum.ibp_forward(weighted_bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)

        # Do not add to lower bound (B = 0 is a lower bound for the noise regions outside the cutoff).
        prob_outside = self.prob_outside()
        corrected_bounds = IntervalBounds(
            bounds.region,
            total_bounds.lower,
            total_bounds.upper + self.barrier_clamp * prob_outside
        )

        return corrected_bounds

    def prob_outside(self):
        # Calculate prob_inside explicitly as it is more numerically accurate compared to summing over probs
        nonzero_scale = torch.count_nonzero(self.scale).item()
        prob_inside = (1.0 - 2 * stats.norm.cdf(-self.sigma_cutoff)) ** nonzero_scale
        prob_outside = 1.0 - prob_inside

        return prob_outside

    def propagate_size(self, in_size):
        out_size1 = self.bound_nominal_dynamics.propagate_size(in_size)
        out_size2 = self.bound_barrier.propagate_size(out_size1)

        assert out_size2 == 1
        return out_size2


class AdditiveGaussianBetaNetwork(Sub):
    def __init__(self, barrier, nominal_dynamics, loc, scale):
        super().__init__(
            AdditiveGaussianExpectation(barrier, nominal_dynamics, loc, scale),
            barrier
        )


class FCNNBarrierNetwork(nn.Sequential):
    activation_class_mapping = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh
    }

    def __init__(self, network_config):
        assert network_config['type'] == 'fcnn'

        activation_class = self.activation_class_mapping[network_config['activation']]

        # First hidden layer (since it requires another input dimension
        layers = [nn.Linear(network_config['input_dim'], network_config['hidden_nodes']), activation_class()]

        # Other hidden layers
        for _ in range(network_config['hidden_layers'] - 1):
            layers.append(nn.Linear(network_config['hidden_nodes'], network_config['hidden_nodes']))
            layers.append(activation_class())

        # Output layer (no activation)
        layers.append(nn.Linear(network_config['hidden_nodes'], 1))

        super().__init__(*layers)


class ResidualBarrierNetwork(nn.Sequential):
    activation_class_mapping = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh
    }

    def __init__(self, network_config):
        assert network_config['type'] == 'residual'

        activation_class = self.activation_class_mapping[network_config['activation']]

        # First hidden layer (since it requires another input dimension
        layers = [nn.Linear(network_config['input_dim'], network_config['hidden_nodes']), activation_class()]

        # Other hidden layers
        for _ in range(network_config['hidden_layers'] - 1):
            layers.append(
                Residual(nn.Sequential(
                    nn.Linear(network_config['hidden_nodes'], network_config['hidden_nodes']),
                    activation_class()
                ))
            )

        # Output layer (no activation)
        layers.append(nn.Linear(network_config['hidden_nodes'], 1))

        super().__init__(*layers)
