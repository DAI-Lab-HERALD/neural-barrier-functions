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

    def clear_relaxation(self):
        self.bound_update.clear_relaxation()

    def backward_relaxation(self, region):
        assert self.bound_update.need_relaxation

        return self.bound_update.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        dt_bounds = LinearBounds(
            linear_bounds.region,
            (linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None
        )
        update_bounds = self.bound_update.crown_backward(dt_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (linear_bounds.lower[0] + self.module.dt * update_bounds.lower[0], linear_bounds.lower[1] + self.module.dt * update_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (linear_bounds.upper[0] + self.module.dt * update_bounds.upper[0], linear_bounds.upper[1] + self.module.dt * update_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        update_bounds = self.bound_update.ibp_forward(bounds, save_relaxation=save_relaxation)
        lower = bounds.lower + self.module.dt * update_bounds.lower
        upper = bounds.upper + self.module.dt * update_bounds.upper

        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        # While we know that in_size and out_size are the same, we need to propagate through bound_update{i} as
        # they may need to store their in and out sizes.
        out_size = self.bound_update.propagate_size(in_size)

        assert in_size == out_size

        return out_size


class RK4(nn.Module):
    def __init__(self, update, dt):
        super().__init__()

        self.update = update
        self.dt = dt

    def forward(self, x):
        k1 = self.dt * self.update(x)
        k2 = self.dt * self.update(x + k1 / 2)
        k3 = self.dt * self.update(x + k2 / 2)
        k4 = self.dt * self.update(x + k3)

        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


class BoundRK4(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_update1 = factory.build(module.update)
        self.bound_update2 = factory.build(module.update)
        self.bound_update3 = factory.build(module.update)
        self.bound_update4 = factory.build(module.update)

    @property
    def need_relaxation(self):
        return self.bound_update1.need_relaxation or self.bound_update2.need_relaxation or \
               self.bound_update3.need_relaxation or self.bound_update4.need_relaxation

    def clear_relaxation(self):
        self.bound_update1.clear_relaxation()
        self.bound_update2.clear_relaxation()
        self.bound_update3.clear_relaxation()
        self.bound_update4.clear_relaxation()

    def backward_relaxation(self, region):
        if self.bound_update1.need_relaxation:
            return self.bound_update1.backward_relaxation(region)
        elif self.bound_update2.need_relaxation:
            update_bounds2, relaxation_module = self.bound_update2.backward_relaxation(region)
            dt_bounds2 = LinearBounds(
                update_bounds2.region,
                (self.module.dt * update_bounds2.lower[0], torch.zeros_like(update_bounds2.lower[1])),
                (self.module.dt * update_bounds2.upper[0], torch.zeros_like(update_bounds2.upper[1]))
            )

            update_bounds1 = self.bound_update1.crown_backward(dt_bounds2)
            return LinearBounds(
                update_bounds1.region,
                (update_bounds2.lower[0] + self.module.dt * update_bounds1.lower[0], update_bounds2.lower[1] + self.module.dt * update_bounds1.lower[1]),
                (update_bounds2.upper[0] + self.module.dt * update_bounds1.upper[0], update_bounds2.upper[1] + self.module.dt * update_bounds1.upper[1])
            ), relaxation_module
        elif self.bound_update3.need_relaxation:
            update_bounds3, relaxation_module = self.bound_update3.backward_relaxation(region)
            dt_bounds3 = LinearBounds(
                update_bounds3.region,
                (self.module.dt * update_bounds3.lower[0] / 2, torch.zeros_like(update_bounds3.lower[1])),
                (self.module.dt * update_bounds3.upper[0] / 2, torch.zeros_like(update_bounds3.upper[1]))
            )

            update_bounds2 = self.bound_update2.crown_backward(dt_bounds3)
            dt_bounds2 = LinearBounds(
                update_bounds2.region,
                (update_bounds3.lower[0] + self.module.dt * update_bounds2.lower[0], torch.zeros_like(update_bounds2.lower[1])),
                (update_bounds3.upper[0] + self.module.dt * update_bounds2.upper[0], torch.zeros_like(update_bounds2.upper[1]))
            )

            update_bounds1 = self.bound_update1.crown_backward(dt_bounds2)
            return LinearBounds(
                update_bounds1.region,
                (update_bounds3.lower[0] + self.module.dt * (update_bounds1.lower[0] + 2 * update_bounds2.lower[0]) / 3, update_bounds3.lower[1] + self.module.dt * (update_bounds1.lower[1] + 2 * update_bounds2.lower[1]) / 3),
                (update_bounds3.upper[0] + self.module.dt * (update_bounds1.upper[0] + 2 * update_bounds2.upper[0]) / 3, update_bounds3.upper[1] + self.module.dt * (update_bounds1.upper[1] + 2 * update_bounds2.upper[1]) / 3)
            ), relaxation_module
        else:
            assert self.bound_update4.need_relaxation
            update_bounds4, relaxation_module = self.bound_update4.backward_relaxation(region)
            dt_bounds4 = LinearBounds(
                update_bounds4.region,
                (self.module.dt * update_bounds4.lower[0] / 2, torch.zeros_like(update_bounds4.lower[1])),
                (self.module.dt * update_bounds4.upper[0] / 2, torch.zeros_like(update_bounds4.upper[1]))
            )

            update_bounds3 = self.bound_update3.crown_backward(dt_bounds4)
            dt_bounds3 = LinearBounds(
                update_bounds3.region,
                (update_bounds4.lower[0] + self.module.dt * update_bounds3.lower[0] / 2, torch.zeros_like(update_bounds3.lower[1])),
                (update_bounds4.lower[0] + self.module.dt * update_bounds3.upper[0] / 2, torch.zeros_like(update_bounds3.upper[1]))
            )

            update_bounds2 = self.bound_update2.crown_backward(dt_bounds3)
            dt_bounds2 = LinearBounds(
                update_bounds2.region,
                (update_bounds4.lower[0] + self.module.dt * update_bounds2.lower[0], torch.zeros_like(update_bounds2.lower[1])),
                (update_bounds4.upper[0] + self.module.dt * update_bounds2.upper[0], torch.zeros_like(update_bounds2.upper[1]))
            )

            update_bounds1 = self.bound_update1.crown_backward(dt_bounds2)
            return LinearBounds(
                update_bounds1.region,
                (update_bounds4.lower[0] + self.module.dt * (update_bounds1.lower[0] + 2 * update_bounds2.lower[0] + 2 * update_bounds3.lower[0]) / 5, update_bounds4.lower[1] + self.module.dt * (update_bounds1.lower[1] + 2 * update_bounds2.lower[1] + 2 * update_bounds3.lower[1]) / 5),
                (update_bounds4.upper[0] + self.module.dt * (update_bounds1.upper[0] + 2 * update_bounds2.upper[0] + 2 * update_bounds3.upper[0]) / 5, update_bounds4.upper[1] + self.module.dt * (update_bounds1.upper[1] + 2 * update_bounds2.upper[1] + 2 * update_bounds3.upper[1]) / 5)
            ), relaxation_module

    def crown_backward(self, linear_bounds):
        dt_bounds4 = LinearBounds(
            linear_bounds.region,
            (linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None
        )
        update_bounds4 = self.bound_update3.crown_backward(dt_bounds4)

        dt_bounds3 = LinearBounds(
            linear_bounds.region,
            (linear_bounds.lower[0] + self.module.dt * update_bounds4.lower[0] / 2, torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (linear_bounds.upper[0] + self.module.dt * update_bounds4.upper[0] / 2, torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None
        )
        update_bounds3 = self.bound_update3.crown_backward(dt_bounds3)

        dt_bounds2 = LinearBounds(
            linear_bounds.region,
            (linear_bounds.lower[0] + self.module.dt * update_bounds3.lower[0] / 2, torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (linear_bounds.upper[0] + self.module.dt * update_bounds3.upper[0] / 2, torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None
        )
        update_bounds2 = self.bound_update2.crown_backward(dt_bounds2)

        dt_bounds1 = LinearBounds(
            linear_bounds.region,
            (linear_bounds.lower[0] + self.module.dt * update_bounds2.lower[0], torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (linear_bounds.upper[0] + self.module.dt * update_bounds2.upper[0], torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None
        )
        update_bounds1 = self.bound_update1.crown_backward(dt_bounds1)

        if linear_bounds.lower is None:
            lower = None
        else:
            lowerA = (update_bounds1.lower[0] + 2 * update_bounds2.lower[0] + 2 * update_bounds3.lower[0] + update_bounds4.lower[0]) / 6
            lowerb = (update_bounds1.lower[1] + 2 * update_bounds2.lower[1] + 2 * update_bounds3.lower[1] + update_bounds4.lower[1]) / 6
            lower = (linear_bounds.lower[0] + self.module.dt * lowerA, linear_bounds.lower[1] + self.module.dt * lowerb)

        if linear_bounds.upper is None:
            upper = None
        else:
            upperA = (update_bounds1.upper[0] + 2 * update_bounds2.upper[0] + 2 * update_bounds3.upper[0] + update_bounds4.upper[0]) / 6
            upperb = (update_bounds1.upper[1] + 2 * update_bounds2.upper[1] + 2 * update_bounds3.upper[1] + update_bounds4.upper[1]) / 6
            upper = (linear_bounds.upper[0] + self.module.dt * upperA, linear_bounds.upper[1] + self.module.dt * upperb)

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        k1 = self.bound_update1.ibp_forward(bounds, save_relaxation=save_relaxation)
        k1 = self.module.dt * k1.lower, self.module.dt * k1.upper

        k2_input_bounds = IntervalBounds(bounds.region, bounds.lower + k1[0] / 2, bounds.upper + k1[1] / 2)
        k2 = self.bound_update2.ibp_forward(k2_input_bounds, save_relaxation=save_relaxation)
        k2 = self.module.dt * k2.lower, self.module.dt * k2.upper

        k3_input_bounds = IntervalBounds(bounds.region, bounds.lower + k2[0] / 2, bounds.upper + k2[1] / 2)
        k3 = self.bound_update3.ibp_forward(k3_input_bounds, save_relaxation=save_relaxation)
        k3 = self.module.dt * k3.lower, self.module.dt * k3.upper

        k4_input_bounds = IntervalBounds(bounds.region, bounds.lower + k3[0], bounds.upper + k3[1])
        k4 = self.bound_update4.ibp_forward(k4_input_bounds, save_relaxation=save_relaxation)
        k4 = self.module.dt * k4.lower, self.module.dt * k4.upper

        lower = bounds.lower + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        upper = bounds.upper + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        # While we know that in_size and out_size are the same, we need to propagate through bound_update{i} as
        # they may need to store their in and out sizes.
        out_size1 = self.bound_update1.propagate_size(in_size)
        out_size2 = self.bound_update2.propagate_size(out_size1)
        out_size3 = self.bound_update3.propagate_size(out_size2)
        out_size4 = self.bound_update4.propagate_size(out_size3)

        assert in_size == out_size1
        assert in_size == out_size2
        assert in_size == out_size3
        assert in_size == out_size4

        return out_size4
