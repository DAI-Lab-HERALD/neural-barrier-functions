from functools import partial

import torch
from bound_propagation import HyperRectangle, BoundModule, LinearBounds, IntervalBounds, BoundModelFactory
from torch import nn


class Affine:
    def __init__(self, A, b, lower, upper):
        self.A, self.b = A, b
        self.lower, self.upper = lower, upper

    def __sub__(self, other):
        if isinstance(other, Affine):
            return Affine(self.A - other.A, self.b - other.b, self.lower, self.upper)
        else:
            return Affine(self.A, self.b - other, self.lower, self.upper)

    def __rsub__(self, other):
        return Affine(-self.A, other - self.b, self.lower, self.upper)

    def __truediv__(self, other):
        return Affine(self.A / other, self.b / other, self.lower, self.upper)

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    @property
    def width(self):
        return self.upper - self.lower

    def partition_max(self):
        if isinstance(self.A, float):
            assert self.A == 0.0
            return self.b

        center, diff = self.center, self.width / 2
        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        weight = self.A.transpose(-1, -2)
        max = center.matmul(weight) + diff.matmul(weight.abs())
        max = max.squeeze(-2) + self.b

        return max

    def partition_min(self):
        if isinstance(self.A, float):
            assert self.A == 0.0
            return self.b

        center, diff = self.center, self.width / 2
        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        weight = self.A.transpose(-1, -2)
        min = center.matmul(weight) - diff.matmul(weight.abs())
        min = min.squeeze(-2) + self.b

        return min

    def mean(self, dim=None):
        A = 0.0 if isinstance(self.A, float) else self.A.mean(dim=dim)
        b = self.b.mean(dim=dim)

        return Affine(A, b, self.lower, self.upper)


def interval_batching(method, lower, upper, batch_size, bound_lower=True, bound_upper=True, reduce=None, **kwargs):
    if batch_size is None:
        interval_bounds = method(HyperRectangle(lower, upper))
        if reduce is not None:
            interval_bounds = reduce(interval_bounds)

        return Affine(0.0, interval_bounds.lower, lower, upper) if bound_lower else None, Affine(0.0, interval_bounds.upper, lower, upper) if bound_upper else None
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))

        if reduce is None:
            batches = [method(HyperRectangle(*batch)) for i, batch in enumerate(batches)]
        else:
            batches = [reduce(method(HyperRectangle(*batch))) for i, batch in enumerate(batches)]

        out_lower = Affine(0.0, torch.cat([batch.lower for batch in batches], dim=-2), lower, upper) if bound_lower else None
        out_upper = Affine(0.0, torch.cat([batch.upper for batch in batches], dim=-2), lower, upper) if bound_upper else None

        return out_lower, out_upper


def linear2interval_batching(method, lower, upper, batch_size, bound_lower=True, bound_upper=True, reduce=None, **kwargs):
    if batch_size is None:
        linear_bounds = method(HyperRectangle(lower, upper))
        if reduce is not None:
            linear_bounds = reduce(linear_bounds)

        interval_bounds = linear_bounds.concretize()

        return Affine(0.0, interval_bounds.lower, lower, upper) if bound_lower else None, Affine(0.0, interval_bounds.upper, lower, upper) if bound_upper else None
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))
        if reduce is None:
            batches = [method(HyperRectangle(*batch)).concretize() for i, batch in enumerate(batches)]
        else:
            batches = [reduce(method(HyperRectangle(*batch))).concretize() for i, batch in enumerate(batches)]

        out_lower = Affine(0.0, torch.cat([batch.lower for batch in batches], dim=-2), lower, upper) if bound_lower else None
        out_upper = Affine(0.0, torch.cat([batch.upper for batch in batches], dim=-2), lower, upper) if bound_upper else None

        return out_lower, out_upper


def linear_batching(method, lower, upper, batch_size, bound_lower=True, bound_upper=True, reduce=None, **kwargs):
    if batch_size is None:
        linear_bounds = method(HyperRectangle(lower, upper))
        if reduce is not None:
            linear_bounds = reduce(linear_bounds)

        return Affine(*linear_bounds.lower, lower, upper) if bound_lower else None, Affine(*linear_bounds.upper, lower, upper) if bound_upper else None
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))
        if reduce is None:
            batches = [method(HyperRectangle(*batch)) for i, batch in enumerate(batches)]
        else:
            batches = [reduce(method(HyperRectangle(*batch))) for i, batch in enumerate(batches)]

        out_lower = Affine(torch.cat([batch.lower[0] for batch in batches], dim=-3), torch.cat([batch.lower[1] for batch in batches], dim=-2), lower, upper) if bound_lower else None
        out_upper = Affine(torch.cat([batch.upper[0] for batch in batches], dim=-3), torch.cat([batch.upper[1] for batch in batches], dim=-2), lower, upper) if bound_upper else None

        return out_lower, out_upper


def bounds(model, partitions, bound_lower=True, bound_upper=True, method='ibp', batch_size=None, **kwargs):
    lower, upper = partitions.lower, partitions.upper

    if method == 'crown_interval':
        method = partial(linear2interval_batching, model.crown, bound_lower=bound_lower, bound_upper=bound_upper)
    elif method == 'crown_linear':
        method = partial(linear_batching, model.crown, bound_lower=bound_lower, bound_upper=bound_upper)
    elif method == 'crown_ibp_interval':
        method = partial(linear2interval_batching, model.crown_ibp, bound_lower=bound_lower, bound_upper=bound_upper)
    elif method == 'crown_ibp_linear':
        method = partial(linear_batching, model.crown_ibp, bound_lower=bound_lower, bound_upper=bound_upper)
    elif method == 'ibp':
        method = partial(interval_batching, model.ibp)
    else:
        raise NotImplementedError()

    return method(lower, upper, batch_size, bound_lower=bound_lower, bound_upper=bound_upper, **kwargs)


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


class LearnedCBFBoundModelFactory(BoundModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.register(Mean, BoundMean)
