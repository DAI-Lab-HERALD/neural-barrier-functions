import torch
from bound_propagation import BoundModule, IntervalBounds, LinearBounds
from torch import nn


class Euler(nn.Module):
    def __init__(self, update, dt):
        super().__init__()

        self.update = update
        self.dt = dt

    def forward(self, x):
        return x + self.dt * self.update(x)


class BoundEuler(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_update = factory.build(module.update)

    @property
    def need_relaxation(self):
        return self.bound_update.need_relaxation

    def crown_backward(self, linear_bounds):
        dt_bounds = LinearBounds(
            linear_bounds.region,
            (linear_bounds.lower[0] / self.module.dt, torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (linear_bounds.upper[0] / self.module.dt, torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None
        )

        update_bounds = self.bound_update.crown_backward(dt_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (linear_bounds.lower[0] + update_bounds.lower[0], linear_bounds.lower[1] + update_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (linear_bounds.upper[0] + update_bounds.upper[0], linear_bounds.upper[1] + update_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        update_bounds = self.bound_update.ibp_forward(bounds, save_relaxation=save_relaxation)
        lower = bounds.lower + self.module.dt * update_bounds.lower
        upper = bounds.upper + self.module.dt * update_bounds.upper

        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        return self.bound_update.propagate_size(in_size)
