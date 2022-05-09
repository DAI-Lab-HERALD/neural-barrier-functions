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


class GaussianProbabilityNetwork(nn.Module):
    def __init__(self, dynamics, loc, scale):
        super().__init__()

        self.dynamics = dynamics
        self.loc, self.scale = loc, scale


class BoundGaussianProbabilityNetwork(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)
        self.out_size = None

        self.bound_dynamics = factory.build(module.dynamics)

    @property
    def need_relaxation(self):
        return self.bound_dynamics.need_relaxation

    def clear_relaxation(self):
        self.bound_dynamics.clear_relaxation()

    def backward_relaxation(self, region):
        return self.bound_dynamics.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        v_region = self.v_region(linear_bounds.region)
        probability = self.probability(v_region)

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

    def v_region(self, region):
        return HyperRectangle(region.lower[..., self.out_size:], region.upper[..., self.out_size:])

    def probability(self, v_region):
        scale_center_erf = lambda v: torch.erf((v - self.module.loc) / (math.sqrt(2) * self.module.scale))
        probability = (scale_center_erf(v_region.upper) - scale_center_erf(v_region.lower)) / 2

        zero_scale = (self.module.scale == 0)
        probability[..., zero_scale] = 1.0

        return probability.prod(dim=-1, keepdim=True)

    def ibp_forward(self, bounds, save_relaxation=False):
        raise NotImplementedError()

    def propagate_size(self, in_size):
        assert in_size % 2 == 0
        self.out_size = in_size // 2
        return self.out_size


class GaussianExpectationRegion(nn.Module):
    def __init__(self, dynamics, loc, scale):
        super().__init__()

        self.dynamics = dynamics
        self.loc, self.scale = loc, scale


class BoundGaussianExpectationRegion(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)
        self.out_size = None

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds):
        v_region = self.v_region(linear_bounds.region)
        expectation = self.expectation(v_region)

        if linear_bounds.lower is None:
            lower = None
        else:
            lowerA = torch.zeros_like(linear_bounds.lower[0]).expand(*[-1 for _ in range(linear_bounds.lower[0].dim() - 1)], linear_bounds.lower[0].size(-1) * 2)
            lower = (lowerA, linear_bounds.lower[0].matmul(expectation.unsqueeze(-1)).squeeze(-1))

        if linear_bounds.upper is None:
            upper = None
        else:
            upperA = torch.zeros_like(linear_bounds.upper[0]).expand(*[-1 for _ in range(linear_bounds.upper[0].dim() - 1)], linear_bounds.upper[0].size(-1) * 2)
            upper = (upperA, linear_bounds.upper[0].matmul(expectation.unsqueeze(-1)).squeeze(-1))

        return LinearBounds(linear_bounds.region, lower, upper)

    def v_region(self, region):
        return HyperRectangle(region.lower[..., self.out_size:], region.upper[..., self.out_size:])

    def expectation(self, v_region):
        scale_center_erf = lambda v: torch.erf((self.module.loc - v) / (math.sqrt(2) * self.module.scale))
        cdf_adjusted_mean = (self.module.loc / 2) * (scale_center_erf(v_region.lower) - scale_center_erf(v_region.upper))

        pdf_exp = lambda v: torch.exp((v - self.module.loc) ** 2 / (-2 * (self.module.scale ** 2)))
        variance_adjustment = (self.module.scale / (math.sqrt(2 * math.pi))) * (pdf_exp(v_region.lower) - pdf_exp(v_region.upper))

        expectation = cdf_adjusted_mean + variance_adjustment

        zero_scale = (self.module.scale == 0)
        expectation[..., zero_scale] = self.module.loc[zero_scale]

        return expectation

    def ibp_forward(self, bounds, save_relaxation=False):
        raise NotImplementedError()

    def propagate_size(self, in_size):
        assert in_size % 2 == 0
        self.out_size = in_size // 2
        return self.out_size


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


class AdditiveGaussianBetaNetwork(Sub):
    def __init__(self, barrier, dynamics, loc, scale):
        super().__init__(
            AdditiveGaussianExpectation(barrier, dynamics, loc, scale),
            barrier
        )

        self.dynamics = dynamics
        self.loc, self.scale = loc, scale


class BoundAdditiveGaussianBetaNetwork(BoundSub):
    def __init__(self, module, factory, state_space_bounds, slices, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_dynamics = factory.build(module.dynamics)
        self.state_space_bounds = state_space_bounds
        self.slices = slices

    def crown_with_relaxation(self, relax, region, bound_lower, bound_upper):
        region = self.partition_noise(region)
        return super().crown_with_relaxation(relax, region, bound_lower, bound_upper)

    def partition_noise(self, region):
        dynamics_bounds = self.bound_dynamics.crown(region).concretize()
        v_min = self.state_space_bounds[0] - dynamics_bounds.upper
        v_max = self.state_space_bounds[1] - dynamics_bounds.lower

        centers = []
        half_widths = []

        for i, num_slices in enumerate(self.slices):
            if self.module.scale[i] == 0.0:
                center = torch.tensor([self.module.loc[i]], device=v_min.device)
                half_width = torch.tensor(0.0, device=v_min.device)
            else:
                v_space = self.vector_linspace(v_min[..., i], v_max[..., i], num_slices).squeeze(-1)
                center = (v_space[:-1] + v_space[1:]) / 2
                half_width = (v_space[1] - v_space[0]) / 2

            centers.append(center)
            half_widths.append(half_width)

        centers = torch.cartesian_prod(*centers)
        half_widths = torch.stack(half_widths, dim=-1)

        lower_v, upper_v = (centers - half_widths).unsqueeze(1).expand(-1, len(region), -1), (centers + half_widths).unsqueeze(1).expand(-1, len(region), -1)

        region_lower, region_upper = region.lower.unsqueeze(0).expand_as(lower_v), region.upper.unsqueeze(0).expand_as(upper_v)

        return HyperRectangle(torch.cat([region_lower, lower_v], dim=-1), torch.cat([region_upper, upper_v], dim=-1))

    def vector_linspace(self, start, stop, steps):
        steps = torch.linspace(0.0, 1.0, steps, device=start.device)
        out = start + steps.unsqueeze(-1) * (stop - start)

        return out


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
