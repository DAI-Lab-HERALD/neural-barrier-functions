from functools import partial

import torch
from bound_propagation import crown, crown_ibp, ibp
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


def interval_batching(method, lower, upper, batch_size, bound_lower=True, bound_upper=True):
    if batch_size is None:
        out_lower, out_upper = method(lower, upper)
        return Affine(0.0, out_lower, lower, upper) if bound_lower else None, Affine(0.0, out_upper, lower, upper) if bound_upper else None
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))
        batches = [method(lower, upper) for (lower, upper) in batches]

        lower = Affine(0.0, torch.cat([batch[0] for batch in batches], dim=-2), lower, upper) if bound_lower else None
        upper = Affine(0.0, torch.cat([batch[1] for batch in batches], dim=-2), lower, upper) if bound_upper else None

        return lower, upper


def linear_batching(method, lower, upper, batch_size, bound_lower=True, bound_upper=True):
    if batch_size is None:
        out_lower, out_upper = method(lower, upper, bound_lower=bound_lower, bound_upper=bound_upper)
        return Affine(*out_lower, lower, upper) if bound_lower else None, Affine(*out_upper, lower, upper) if bound_upper else None
    else:
        batches = list(zip(lower.split(batch_size), upper.split(batch_size)))
        batches = [method(lower, upper, bound_lower=bound_lower, bound_upper=bound_upper) for (lower, upper) in batches]

        lower = Affine(torch.cat([batch[0][0] for batch in batches], dim=-2), torch.cat([batch[0][1] for batch in batches], dim=-2), lower, upper) if bound_lower else None
        upper = Affine(torch.cat([batch[1][0] for batch in batches], dim=-2), torch.cat([batch[1][1] for batch in batches], dim=-2), lower, upper) if bound_upper else None

        return lower, upper


class Barrier(nn.Sequential):
    def interval(self, partitions, prefix=None, bound_lower=True, bound_upper=True, method='crown_ibp_linear', batch_size=None, **kwargs):
        if prefix is None:
            lower, upper = partitions.lower, partitions.upper
            model = self
        else:
            lower, upper = partitions.lower, partitions.upper
            model = nn.Sequential(*list(prefix.children()), *list(self.children()))

        if method == 'crown_interval':
            model = crown(model)
            method = partial(interval_batching, model.crown_interval)
        elif method == 'crown_linear':
            model = crown(model)
            method = partial(linear_batching, model.crown_linear)
        elif method == 'crown_ibp_interval':
            model = crown_ibp(model)
            method = partial(interval_batching, model.crown_ibp_interval)
        elif method == 'crown_ibp_linear':
            model = crown_ibp(model)
            method = partial(linear_batching, model.crown_ibp_linear)
        elif method == 'ibp':
            model = ibp(model)
            method = partial(interval_batching, model.ibp)
        else:
            raise NotImplementedError()

        return method(lower, upper, batch_size, bound_lower=bound_lower, bound_upper=bound_upper)
