import torch
from bound_propagation import Residual, BoundModule, LinearBounds, IntervalBounds, HyperRectangle
from torch import nn


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

    def combined(self, set):
        region = HyperRectangle(set.lower, set.upper)

        expectation = self.bound_dynamics_barrier.crown_ibp(region, bound_lower=False).concretize()
        barrier = self.bound_barrier.ibp(region)

        return expectation.upper - barrier.lower


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
