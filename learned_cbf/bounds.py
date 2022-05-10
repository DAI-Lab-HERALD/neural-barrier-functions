from functools import partial

import torch
from bound_propagation import HyperRectangle, BoundModule, LinearBounds, IntervalBounds, BoundModelFactory
from torch import nn

from .networks import BoundBetaNetwork, BetaNetwork, Mean, BoundMean, BoundSum, Sum, BoundGaussianExpectationRegion, \
    GaussianExpectationRegion, BoundGaussianProbabilityNetwork, GaussianProbabilityNetwork, DynamicsNoise, \
    BoundDynamicsNoise, BoundAdditiveGaussianExpectation, AdditiveGaussianExpectation


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

    @property
    def volume(self):
        return self.width.prod(dim=-1)

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
    if batch_size is None:
        interval_bounds = method(HyperRectangle(lower, upper))

        return Affine(0.0, interval_bounds.lower, lower, upper) if bound_lower else None, Affine(0.0, interval_bounds.upper, lower, upper) if bound_upper else None
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))
        batches = [method(HyperRectangle(*batch)) for i, batch in enumerate(batches)]

        out_lower = Affine(0.0, torch.cat([batch.lower for batch in batches], dim=-2), lower, upper) if bound_lower else None
        out_upper = Affine(0.0, torch.cat([batch.upper for batch in batches], dim=-2), lower, upper) if bound_upper else None

        return out_lower, out_upper


def linear2interval_batching(method, lower, upper, batch_size, bound_lower=True, bound_upper=True, **kwargs):
    if batch_size is None:
        interval_bounds = method(HyperRectangle(lower, upper), bound_lower=bound_lower, bound_upper=bound_upper).concretize()

        return Affine(0.0, interval_bounds.lower, lower, upper) if bound_lower else None, Affine(0.0, interval_bounds.upper, lower, upper) if bound_upper else None
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))
        batches = [method(HyperRectangle(*batch), bound_lower=bound_lower, bound_upper=bound_upper).concretize() for i, batch in enumerate(batches)]

        out_lower = Affine(0.0, torch.cat([batch.lower for batch in batches], dim=-2), lower, upper) if bound_lower else None
        out_upper = Affine(0.0, torch.cat([batch.upper for batch in batches], dim=-2), lower, upper) if bound_upper else None

        return out_lower, out_upper


def linear_batching(method, lower, upper, batch_size, bound_lower=True, bound_upper=True, **kwargs):
    if batch_size is None:
        linear_bounds = method(HyperRectangle(lower, upper), bound_lower=bound_lower, bound_upper=bound_upper)

        return Affine(*linear_bounds.lower, lower, upper) if bound_lower else None, Affine(*linear_bounds.upper, lower, upper) if bound_upper else None
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))
        batches = [method(HyperRectangle(*batch), bound_lower=bound_lower, bound_upper=bound_upper) for i, batch in enumerate(batches)]

        out_lower = Affine(torch.cat([batch.lower[0] for batch in batches], dim=-3), torch.cat([batch.lower[1] for batch in batches], dim=-2), lower, upper) if bound_lower else None
        out_upper = Affine(torch.cat([batch.upper[0] for batch in batches], dim=-3), torch.cat([batch.upper[1] for batch in batches], dim=-2), lower, upper) if bound_upper else None

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


class LearnedCBFBoundModelFactory(BoundModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.register(Mean, BoundMean)
        self.register(Sum, BoundSum)
        self.register(BetaNetwork, BoundBetaNetwork)
        self.register(GaussianProbabilityNetwork, BoundGaussianProbabilityNetwork)
        self.register(GaussianExpectationRegion, BoundGaussianExpectationRegion)
        self.register(DynamicsNoise, BoundDynamicsNoise)
        self.register(AdditiveGaussianExpectation, BoundAdditiveGaussianExpectation)
