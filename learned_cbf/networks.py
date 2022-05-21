import copy
import math

import torch
from bound_propagation import Residual, BoundModule, LinearBounds, IntervalBounds, HyperRectangle, Sub, BoundActivation, \
    Clamp, Parallel, Add, BoundSub
from bound_propagation.activation import assert_bound_order, BoundSigmoid, crown_backward_act_jit
from bound_propagation.linear import crown_backward_linear_jit
from torch import nn, Tensor


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

    def crown_backward(self, linear_bounds):
        subnetwork_bounds = self.subnetwork.crown_backward(linear_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (subnetwork_bounds.lower[0].mean(dim=0), subnetwork_bounds.lower[1].mean(dim=0))

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (subnetwork_bounds.upper[0].mean(dim=0), subnetwork_bounds.upper[1].mean(dim=0))

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        subnetwork_bounds = self.subnetwork.ibp_forward(bounds, save_relaxation=save_relaxation)
        return IntervalBounds(bounds.region, subnetwork_bounds.lower.mean(dim=0), subnetwork_bounds.upper.mean(dim=0))

    def propagate_size(self, in_size):
        return self.subnetwork.propagate_size(in_size)


class BetaNetwork(nn.Module):
    def __init__(self, dynamics, barrier):
        super().__init__()

        self.dynamics_barrier = Mean(nn.Sequential(dynamics, barrier))
        self.barrier = barrier

    def forward(self, x):
        return self.dynamics_barrier(x) - self.barrier(x)


class BoundBetaNetwork(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_dynamics_barrier = factory.build(module.dynamics_barrier)
        self.bound_barrier = factory.build(module.barrier)

    @property
    def need_relaxation(self):
        return self.bound_dynamics_barrier.need_relaxation or self.bound_barrier.need_relaxation

    def clear_relaxation(self):
        self.bound_dynamics_barrier.clear_relaxation()
        self.bound_barrier.clear_relaxation()

    def backward_relaxation(self, region):
        if self.bound_dynamics_barrier.need_relaxation:
            return self.bound_dynamics_barrier.backward_relaxation(region)
        else:
            assert self.bound_barrier.need_relaxation
            return self.bound_barrier.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        input_bounds = LinearBounds(
            linear_bounds.region,
            (linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None,
        )
        linear_bounds1 = self.bound_dynamics_barrier.crown_backward(input_bounds)

        # We can only do this inversion for a "Sub" module because we know nothing is ahead (after in the network).
        input_bounds = LinearBounds(
            linear_bounds.region,
            (linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None,
            (linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
        )
        linear_bounds2 = self.bound_barrier.crown_backward(input_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (linear_bounds1.lower[0] - linear_bounds2.upper[0], linear_bounds1.lower[1] - linear_bounds2.upper[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (linear_bounds1.upper[0] - linear_bounds2.lower[0], linear_bounds1.upper[1] - linear_bounds2.lower[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        bounds1 = self.bound_dynamics_barrier.ibp_forward(bounds, save_relaxation=save_relaxation)
        bounds2 = self.bound_barrier.ibp_forward(bounds, save_relaxation=save_relaxation)

        return IntervalBounds(
            bounds.region,
            bounds1.lower - bounds2.upper,
            bounds1.upper - bounds2.lower
        )

    def propagate_size(self, in_size):
        out_size1 = self.bound_dynamics_barrier.propagate_size(in_size)
        out_size2 = self.bound_barrier.propagate_size(in_size)

        assert out_size1 == out_size2

        return out_size1


class Sum(nn.Module):
    def __init__(self, subnetwork):
        super().__init__()

        self.subnetwork = subnetwork

    def forward(self, x):
        return self.subnetwork(x).mean(dim=0)


class BoundSum(BoundModule):
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

    def crown_backward(self, linear_bounds):
        subnetwork_bounds = self.subnetwork.crown_backward(linear_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (subnetwork_bounds.lower[0].sum(dim=0), subnetwork_bounds.lower[1].sum(dim=0))

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (subnetwork_bounds.upper[0].sum(dim=0), subnetwork_bounds.upper[1].sum(dim=0))

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        subnetwork_bounds = self.subnetwork.ibp_forward(bounds, save_relaxation=save_relaxation)
        return IntervalBounds(bounds.region, subnetwork_bounds.lower.sum(dim=0), subnetwork_bounds.upper.sum(dim=0))

    def propagate_size(self, in_size):
        return self.subnetwork.propagate_size(in_size)


class VRegionMixin:
    def v_region(self, region):
        loc, scale = self.module.loc.to(region.lower.device), self.module.scale.to(region.lower.device)

        linear_bounds = self.initial_linear_bounds(region, self.state_size)
        dynamics_bounds = self.bound_dynamics.crown_backward(linear_bounds).concretize()
        v_min = torch.max(self.state_space_bounds[0] - dynamics_bounds.upper, loc - self.sigma_cut_off * scale)
        v_max = torch.min(self.state_space_bounds[1] - dynamics_bounds.lower, loc + self.sigma_cut_off * scale)

        centers = []
        half_widths = []

        for i, num_slices in enumerate(self.slices):
            if self.module.scale[i] == 0.0:
                center = torch.full((v_min.size(0), 1), loc[i], device=v_min.device)
                half_width = torch.zeros((v_min.size(0), 1), device=v_min.device)
            else:
                # v_space = vector_linspace(v_min[..., i], v_max[..., i], num_slices + 1).squeeze(-1)
                v_space = gaussian_partitioning(v_min[..., i], v_max[..., i], num_slices + 1, loc[i], scale[i]).squeeze(-1)
                center = (v_space[:, :-1] + v_space[:, 1:]) / 2
                half_width = (v_space[:, 1:] - v_space[:, :-1]) / 2

            centers.append(center)
            half_widths.append(half_width)

        centers = batch_cartesian_prod(*centers).transpose(0, 1)
        half_widths = batch_cartesian_prod(*half_widths).transpose(0, 1)
        lower, upper = centers - half_widths, centers + half_widths

        return HyperRectangle(lower, upper)


def vector_linspace(start, stop, steps):
    start, stop = start.unsqueeze(-1), stop.unsqueeze(-1)
    steps = torch.linspace(0.0, 1.0, steps, device=start.device)
    out = start + steps * (stop - start)

    return out


def gaussian_partitioning(start, stop, steps, loc, scale):
    linspace = vector_linspace(start, stop, steps)

    start, stop = start.unsqueeze(-1), stop.unsqueeze(-1)
    steps = torch.linspace(0.0, 1.0, steps, device=start.device)

    def cdf(x):
        return 0.5 * (1 + torch.erf((x - loc) / (scale * math.sqrt(2))))

    def inverse_cdf(x):
        return loc + scale * math.sqrt(2) * torch.erfinv(2 * x - 1)

    out = inverse_cdf(steps * (cdf(stop) - cdf(start)) + cdf(start))
    out = torch.where(out.isinf(), linspace, out)
    out[:, 0], out[:, -1] = start[:, 0], stop[:, 0]

    return out


def batch_cartesian_prod(*args):
    batch_size = args[0].size(0)
    dim = len(args)
    view = [batch_size] + [1 for _ in range(dim)]
    expand_size = [batch_size] + [arg.size(-1) for arg in args]

    def view_dim(i, arg):
        v = copy.deepcopy(view)
        v[i + 1] = arg.size(-1)
        return v

    args = [arg.view(view_dim(i, arg)).expand(*expand_size) for i, arg in enumerate(args)]
    stacked = torch.stack(args, dim=-1).view(batch_size, -1, dim)
    return stacked


class GaussianProbabilityNetwork(nn.Module):
    def __init__(self, dynamics, loc, scale):
        super().__init__()

        self.dynamics = dynamics
        self.loc, self.scale = loc, scale


class BoundGaussianProbabilityNetwork(BoundModule, VRegionMixin):
    def __init__(self, module, factory, state_space_bounds, slices, sigma_cut_off=10.0, **kwargs):
        super().__init__(module, factory, **kwargs)
        self.state_size = None

        self.bound_dynamics = factory.build(module.dynamics)
        self.state_space_bounds = state_space_bounds
        self.slices = slices
        self.sigma_cut_off = sigma_cut_off

    @property
    def need_relaxation(self):
        return self.bound_dynamics.need_relaxation

    def clear_relaxation(self):
        self.bound_dynamics.clear_relaxation()

    def backward_relaxation(self, region):
        return self.bound_dynamics.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        probability = self.probability(linear_bounds.region)

        subnetworks_bounds = self.bound_dynamics.crown_backward(linear_bounds)

        if subnetworks_bounds.lower is None:
            lower = None
        else:
            lower = (probability.unsqueeze(-1) * subnetworks_bounds.lower[0], probability * subnetworks_bounds.lower[1])

        if subnetworks_bounds.upper is None:
            upper = None
        else:
            upper = (probability.unsqueeze(-1) * subnetworks_bounds.upper[0], probability * subnetworks_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def probability(self, region):
        v_region = self.v_region(region)
        loc, scale = self.module.loc.to(v_region.lower.device), self.module.scale.to(v_region.lower.device)

        def scale_center_erf(v):
            return torch.erf((v - loc) / (math.sqrt(2) * scale))
        probability = (scale_center_erf(v_region.upper) - scale_center_erf(v_region.lower)) / 2

        zero_scale = (scale == 0)
        probability[..., zero_scale] = 1.0

        return probability.prod(dim=-1, keepdim=True)

    def ibp_forward(self, bounds, save_relaxation=False):
        raise NotImplementedError()

    def propagate_size(self, in_size):
        self.state_size = in_size
        out_size = self.bound_dynamics.propagate_size(in_size)
        assert out_size == self.state_size
        return self.state_size


class GaussianExpectationRegion(nn.Module):
    def __init__(self, dynamics, loc, scale):
        super().__init__()

        self.dynamics = dynamics
        self.loc, self.scale = loc, scale


class BoundGaussianExpectationRegion(BoundModule, VRegionMixin):
    def __init__(self, module, factory, state_space_bounds, slices, sigma_cut_off=10.0, **kwargs):
        super().__init__(module, factory, **kwargs)
        self.state_size = None

        self.bound_dynamics = factory.build(module.dynamics)
        self.state_space_bounds = state_space_bounds
        self.slices = slices
        self.sigma_cut_off = sigma_cut_off

    @property
    def need_relaxation(self):
        return self.bound_dynamics.need_relaxation

    def clear_relaxation(self):
        self.bound_dynamics.clear_relaxation()

    def backward_relaxation(self, region):
        return self.bound_dynamics.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        expectation = self.expectation(linear_bounds.region)

        if linear_bounds.lower is None:
            lower = None
        else:
            lowerA = torch.zeros_like(linear_bounds.lower[0])
            if lowerA.dim() == 3:
                lowerA = lowerA.unsqueeze(0).expand(expectation.size(0), -1, -1, -1)
            lower = (lowerA, linear_bounds.lower[0].matmul(expectation.unsqueeze(-1)).squeeze(-1) + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upperA = torch.zeros_like(linear_bounds.upper[0])
            if upperA.dim() == 3:
                upperA = upperA.unsqueeze(0).expand(expectation.size(0), -1, -1, -1)
            upper = (upperA, linear_bounds.upper[0].matmul(expectation.unsqueeze(-1)).squeeze(-1) + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def expectation(self, region):
        v_region = self.v_region(region)
        loc, scale = self.module.loc.to(v_region.lower.device), self.module.scale.to(v_region.lower.device)

        def scale_center_erf(v):
            return torch.erf((loc - v) / (math.sqrt(2) * scale))
        cdf_adjusted_mean = (loc / 2) * (scale_center_erf(v_region.lower) - scale_center_erf(v_region.upper))

        def pdf_exp(v):
            return torch.exp((v - loc) ** 2 / (-2 * (scale ** 2)))
        variance_adjustment = (scale / (math.sqrt(2 * math.pi))) * (pdf_exp(v_region.lower) - pdf_exp(v_region.upper))

        expectation = cdf_adjusted_mean + variance_adjustment

        zero_scale = (scale == 0)
        expectation[..., zero_scale] = loc[zero_scale]

        return expectation

    def ibp_forward(self, bounds, save_relaxation=False):
        raise NotImplementedError()

    def propagate_size(self, in_size):
        self.state_size = in_size
        out_size = self.bound_dynamics.propagate_size(in_size)
        assert out_size == self.state_size
        return self.state_size


class DynamicsNoise(nn.Module):
    def __init__(self, dynamics, loc, scale):
        super().__init__()

        self.dynamics = dynamics
        self.loc, self.scale = loc, scale


class BoundDynamicsNoise(BoundModule, VRegionMixin):
    def __init__(self, module, factory, state_space_bounds, slices, sigma_cut_off=10.0, **kwargs):
        super().__init__(module, factory, **kwargs)
        self.state_size = None

        self.bound_dynamics = factory.build(module.dynamics)
        self.state_space_bounds = state_space_bounds
        self.slices = slices
        self.sigma_cut_off = sigma_cut_off

    @property
    def need_relaxation(self):
        return self.bound_dynamics.need_relaxation

    def clear_relaxation(self):
        self.bound_dynamics.clear_relaxation()

    def backward_relaxation(self, region):
        return self.bound_dynamics.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        assert linear_bounds.lower is not None and linear_bounds.upper is not None

        v_region = self.v_region(linear_bounds.region)
        region_lower, region_upper = linear_bounds.region.lower.unsqueeze(0).expand_as(v_region.lower), \
                                     linear_bounds.region.upper.unsqueeze(0).expand_as(v_region.upper)
        extended_region = HyperRectangle(torch.cat([region_lower, v_region.lower], dim=-1), torch.cat([region_upper, v_region.upper], dim=-1))

        subnetworks_bounds = self.bound_dynamics.crown_backward(linear_bounds)

        lowerA = torch.cat([subnetworks_bounds.lower[0], linear_bounds.lower[0]], dim=-1)
        lower_bias = subnetworks_bounds.lower[1]
        if lowerA.dim() == 3:
            lowerA = lowerA.unsqueeze(0).expand(v_region.lower.size(0), -1, -1, -1)
            lower_bias = lower_bias.unsqueeze(0).expand(v_region.lower.size(0), -1, -1)
        lower = (lowerA, lower_bias)

        upperA = torch.cat([subnetworks_bounds.upper[0], linear_bounds.upper[0]], dim=-1)
        upper_bias = subnetworks_bounds.upper[1]
        if upperA.dim() == 3:
            upperA = upperA.unsqueeze(0).expand(v_region.upper.size(0), -1, -1, -1)
            upper_bias = upper_bias.unsqueeze(0).expand(v_region.upper.size(0), -1, -1)
        upper = (upperA, upper_bias)

        linear_bounds = LinearBounds(extended_region, lower, upper)
        return linear_bounds

    def ibp_forward(self, bounds, save_relaxation=False):
        raise NotImplementedError()

    def propagate_size(self, in_size):
        self.state_size = in_size
        out_size = self.bound_dynamics.propagate_size(in_size)
        assert out_size == self.state_size
        return self.state_size


class AdditiveGaussianExpectation(Sum):
    def __init__(self, barrier, dynamics, loc, scale):
        super().__init__(
            nn.Sequential(
                Add(
                    GaussianProbabilityNetwork(dynamics, loc, scale),
                    GaussianExpectationRegion(dynamics, loc, scale),
                ),
                barrier
            )
        )

        self.dynamics = dynamics
        self.loc, self.scale = loc, scale


class BoundAdditiveGaussianExpectation(BoundSum, VRegionMixin):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.subnetwork = factory.build(module.subnetwork)
        self.barrier_dynamics = factory.build(
            nn.Sequential(
                DynamicsNoise(module.dynamics, module.loc, module.scale),
                module.subnetwork[1]
            )
        )

        self.subnetwork.bound_sequential[1] = self.barrier_dynamics.bound_sequential[1]

    @property
    def need_relaxation(self):
        return self.subnetwork.need_relaxation or self.barrier_dynamics.need_relaxation

    def clear_relaxation(self):
        self.subnetwork.clear_relaxation()
        self.barrier_dynamics.clear_relaxation()

    def backward_relaxation(self, region):
        if self.subnetwork.bound_sequential[0].need_relaxation:
            return self.subnetwork.backward_relaxation(region)

        return self.barrier_dynamics.backward_relaxation(region)

    def propagate_size(self, in_size):
        out_size1 = self.subnetwork.propagate_size(in_size)
        out_size2 = self.barrier_dynamics.propagate_size(in_size)

        assert out_size1 == out_size2
        return out_size1


class AdditiveGaussianBetaNetwork(Sub):
    def __init__(self, barrier, dynamics, loc, scale):
        super().__init__(
            AdditiveGaussianExpectation(barrier, dynamics, loc, scale),
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
