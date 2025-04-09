import torch
from bound_propagation import BoundModule, IntervalBounds, LinearBounds
from torch import nn


class ButcherTableau(nn.Module):
    def __init__(self, update, dt, a, b):
        super().__init__()

        self.update = update
        self.dt = dt

        self.a = a
        self.b = b

        assert all(aij >= 0.0 for ai in a for aij in ai)
        assert all(bi >= 0.0 for bi in b)

    def forward(self, x):
        k = [self.update(x)]

        for ai in self.a:
            weighted_k = (torch.stack(k, dim=-1) * torch.tensor(ai, device=x.device)).sum(dim=-1)
            next_k = self.update(x + self.dt * weighted_k)
            k.append(next_k)

        weighted_k = (torch.stack(k, dim=-1) * torch.tensor(self.b, device=x.device)).sum(dim=-1)
        return x + self.dt * weighted_k


class BoundButcherTableau(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_update = nn.ModuleList([
            factory.build(module.update) for _ in range(len(module.b))
        ])

    @property
    def need_relaxation(self):
        return any(bound_update.need_relaxation for bound_update in self.bound_update)

    def clear_relaxation(self):
        for bound_update in self.bound_update:
            bound_update.clear_relaxation()

    def backward_relaxation(self, region):
        for i, bound_update in enumerate(self.bound_update):
            if bound_update.need_relaxation:
                return self.subnetwork_backward_relaxation(self.bound_update[:i], bound_update, region)

        assert False, 'No substep need relaxation'

    def subnetwork_backward_relaxation(self, network, relaxation_step, region):
        linear_bounds, relaxation_module = relaxation_step.backward_relaxation(region)

        zero_bias = torch.zeros_like(linear_bounds.lower[1])
        update_bounds = [None for _ in network]

        for i, bound_update in reversed(list(enumerate(network))):
            if linear_bounds.lower is None:
                lower = None
            else:
                lowerA = self.module.dt * self.module.a[len(network) - 1][i] * linear_bounds.lower[0]
                for k, bounds in enumerate(network[i + 1:]):
                    j = i + k + 1
                    lowerA = lowerA + self.module.dt * self.module.a[j - 1][i] * update_bounds[j].lower[0]
                lower = (lowerA, zero_bias)

            if linear_bounds.upper is None:
                upper = None
            else:
                upperA = self.module.dt * self.module.a[len(network) - 1][i] * linear_bounds.upper[0]
                for k, bounds in enumerate(network[i + 1:]):
                    j = i + k + 1
                    upperA = upperA + self.module.dt * self.module.a[j - 1][i] * update_bounds[j].upper[0]
                upper = (upperA, zero_bias)

            dt_bounds = LinearBounds(linear_bounds.region, lower, upper)
            bounds = bound_update.crown_backward(dt_bounds, optimize)
            update_bounds[i] = bounds

        if linear_bounds.lower is None:
            lower = None
        else:
            lowerA = linear_bounds.lower[0]
            lower_bias = linear_bounds.lower[1]
            for bounds in update_bounds:
                lowerA = lowerA + bounds.lower[0]
                lower_bias = lower_bias + bounds.lower[1]
            lower = (lowerA, lower_bias)

        if linear_bounds.upper is None:
            upper = None
        else:
            upperA = linear_bounds.upper[0]
            upper_bias = linear_bounds.upper[1]
            for bounds in update_bounds:
                upperA = upperA + bounds.upper[0]
                upper_bias = upper_bias + bounds.upper[1]
            upper = (upperA, upper_bias)

        return LinearBounds(linear_bounds.region, lower, upper), relaxation_module

    def crown_backward(self, linear_bounds, optimize):
        zero_bias = torch.zeros_like(linear_bounds.lower[1] if linear_bounds.lower is not None else linear_bounds.upper[1])
        update_bounds = [None for _ in self.bound_update]

        for i, bound_update in reversed(list(enumerate(self.bound_update))):
            if linear_bounds.lower is None:
                lower = None
            else:
                lowerA = self.module.dt * self.module.b[i] * linear_bounds.lower[0]
                for k, bounds in enumerate(self.bound_update[i + 1:]):
                    j = i + k + 1
                    if self.module.a[j - 1][i] > 0.0:
                        lowerA = lowerA + self.module.dt * self.module.a[j - 1][i] * update_bounds[j].lower[0]
                lower = (lowerA, zero_bias)

            if linear_bounds.upper is None:
                upper = None
            else:
                upperA = self.module.dt * self.module.b[i] * linear_bounds.upper[0]
                for k, bounds in enumerate(self.bound_update[i + 1:]):
                    j = i + k + 1
                    if self.module.a[j - 1][i] > 0.0:
                        upperA = upperA + self.module.dt * self.module.a[j - 1][i] * update_bounds[j].upper[0]
                upper = (upperA, zero_bias)

            dt_bounds = LinearBounds(linear_bounds.region, lower, upper)
            bounds = bound_update.crown_backward(dt_bounds, optimize)
            update_bounds[i] = bounds

        if linear_bounds.lower is None:
            lower = None
        else:
            lowerA = linear_bounds.lower[0]
            lower_bias = linear_bounds.lower[1]
            for bounds in update_bounds:
                lowerA = lowerA + bounds.lower[0]
                lower_bias = lower_bias + bounds.lower[1]
            lower = (lowerA, lower_bias)

        if linear_bounds.upper is None:
            upper = None
        else:
            upperA = linear_bounds.upper[0]
            upper_bias = linear_bounds.upper[1]
            for bounds in update_bounds:
                upperA = upperA + bounds.upper[0]
                upper_bias = upper_bias + bounds.upper[1]
            upper = (upperA, upper_bias)

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, input_bounds, save_relaxation=False, save_input_bounds=False):
        update_bounds = self.bound_update[0].ibp_forward(input_bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)
        lower = [update_bounds.lower]
        upper = [update_bounds.upper]

        for i, ai in enumerate(self.module.a):
            weighted_lower = (torch.stack(lower, dim=-1) * torch.tensor(ai, device=input_bounds.lower.device)).sum(dim=-1)
            weighted_upper = (torch.stack(upper, dim=-1) * torch.tensor(ai, device=input_bounds.lower.device)).sum(dim=-1)

            bounds = IntervalBounds(input_bounds.region,
                                    input_bounds.lower + self.module.dt * weighted_lower,
                                    input_bounds.upper + self.module.dt * weighted_upper)
            next_bounds = self.bound_update[i + 1].ibp_forward(bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)
            lower.append(next_bounds.lower)
            upper.append(next_bounds.upper)

        weighted_lower = (torch.stack(lower, dim=-1) * torch.tensor(self.module.b, device=input_bounds.lower.device)).sum(dim=-1)
        weighted_upper = (torch.stack(upper, dim=-1) * torch.tensor(self.module.b, device=input_bounds.lower.device)).sum(dim=-1)
        return IntervalBounds(
            input_bounds.region,
            input_bounds.lower + self.module.dt * weighted_lower,
            input_bounds.upper + self.module.dt * weighted_upper
        )

    def propagate_size(self, in_size):
        # While we know that in_size and out_size are the same, we need to propagate through bound_update{i} as
        # they may need to store their in and out sizes.
        size = in_size
        for bound_update in self.bound_update:
            size = bound_update.propagate_size(size)
            assert in_size == size

        return size


class Euler(ButcherTableau):
    def __init__(self, update, dt):
        super().__init__(update, dt, [], [1.0])


class Heun(ButcherTableau):
    def __init__(self, update, dt):
        super().__init__(update, dt, [[1.0]], [0.5, 0.5])


class RK4(ButcherTableau):
    def __init__(self, update, dt):
        super().__init__(update, dt, [[0.5], [0.0, 0.5], [0.0, 0.0, 1.0]], [1/6, 1/3, 1/3, 1/6])