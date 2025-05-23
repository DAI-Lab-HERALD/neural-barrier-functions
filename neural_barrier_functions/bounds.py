from functools import partial

import torch
from bound_propagation import HyperRectangle, BoundModule, LinearBounds, IntervalBounds, BoundModelFactory
from torch import nn

from .partitioning import Partitions
from .networks import Mean, BoundMean, BoundSum, Sum, BoundAdditiveGaussianExpectation, AdditiveGaussianExpectation


class Affine:
    def __init__(self, A, b, partitions):
        self.A, self.b = A, b
        self.partitions = partitions

    def __sub__(self, other):
        if isinstance(other, Affine):
            return Affine(self.A - other.A, self.b - other.b, self.partitions)
        else:
            return Affine(self.A, self.b - other, self.partitions)

    def __rsub__(self, other):
        return Affine(-self.A, other - self.b, self.partitions)

    def __truediv__(self, other):
        return Affine(self.A / other, self.b / other, self.partitions)

    @property
    def center(self):
        return self.partitions.center

    @property
    def width(self):
        return self.partitions.width

    @property
    def volume(self):
        return self.width.prod(dim=-1)

    def __getitem__(self, item):
        if isinstance(self.A, float):
            assert self.A == 0.0
            return Affine(0.0, self.b[item], self.partitions[item])

        return Affine(self.A[item], self.b[item], self.partitions[item])

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


def interval_batching(method, lower, upper, batch_size, bound_lower=True, bound_upper=True, **kwargs):
    input_regions = Partitions((lower, upper))

    if batch_size is None:
        interval_bounds = method(HyperRectangle(lower, upper))
        return Affine(0.0, interval_bounds.lower, input_regions) if bound_lower else None, Affine(0.0, interval_bounds.upper, input_regions) if bound_upper else None
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))
        batches = [method(HyperRectangle(*batch)) for i, batch in enumerate(batches)]

        out_lower = Affine(0.0, torch.cat([batch.lower for batch in batches], dim=-2), input_regions) if bound_lower else None
        out_upper = Affine(0.0, torch.cat([batch.upper for batch in batches], dim=-2), input_regions) if bound_upper else None

        return out_lower, out_upper


def linear2interval_batching(method, lower, upper, batch_size, bound_lower=True, bound_upper=True, **kwargs):
    input_regions = Partitions((lower, upper))

    if batch_size is None:
        interval_bounds = method(HyperRectangle(lower, upper), bound_lower=bound_lower, bound_upper=bound_upper).concretize()

        return Affine(0.0, interval_bounds.lower, input_regions) if bound_lower else None, Affine(0.0, interval_bounds.upper, input_regions) if bound_upper else None
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))
        batches = [method(HyperRectangle(*batch), bound_lower=bound_lower, bound_upper=bound_upper).concretize() for i, batch in enumerate(batches)]

        out_lower = Affine(0.0, torch.cat([batch.lower for batch in batches], dim=-2), input_regions) if bound_lower else None
        out_upper = Affine(0.0, torch.cat([batch.upper for batch in batches], dim=-2), input_regions) if bound_upper else None

        return out_lower, out_upper


def linear_batching(method, lower, upper, batch_size, bound_lower=True, bound_upper=True, **kwargs):
    input_regions = Partitions((lower, upper))

    if batch_size is None:
        linear_bounds = method(HyperRectangle(lower, upper), bound_lower=bound_lower, bound_upper=bound_upper)

        out_lower = Affine(*linear_bounds.lower, input_regions) if bound_lower else None
        out_upper = Affine(*linear_bounds.upper, input_regions) if bound_upper else None

        return out_lower, out_upper
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))
        batches = [method(HyperRectangle(*batch), bound_lower=bound_lower, bound_upper=bound_upper) for i, batch in enumerate(batches)]

        out_lower = Affine(torch.cat([batch.lower[0] for batch in batches], dim=-3), torch.cat([batch.lower[1] for batch in batches], dim=-2), input_regions) if bound_lower else None
        out_upper = Affine(torch.cat([batch.upper[0] for batch in batches], dim=-3), torch.cat([batch.upper[1] for batch in batches], dim=-2), input_regions) if bound_upper else None

        return out_lower, out_upper


def bounds(model, partitions, bound_lower=True, bound_upper=True, method='ibp', batch_size=None, **kwargs):
    lower, upper = partitions.lower, partitions.upper

    if method == 'crown_interval':
        method = partial(linear2interval_batching, model.crown)
    elif method == 'crown_linear':
        method = partial(linear_batching, model.crown)
    elif method == 'crown_ibp_interval':
        method = partial(linear2interval_batching, model.crown_ibp)
    elif method == 'crown_ibp_linear':
        method = partial(linear_batching, model.crown_ibp)
    elif method == 'ibp':
        method = partial(interval_batching, model.ibp)
    else:
        raise NotImplementedError()

    return method(lower, upper, batch_size, bound_lower=bound_lower, bound_upper=bound_upper, **kwargs)


class NBFBoundModelFactory(BoundModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.register(Mean, BoundMean)
        self.register(Sum, BoundSum)
        self.register(AdditiveGaussianExpectation, BoundAdditiveGaussianExpectation)
