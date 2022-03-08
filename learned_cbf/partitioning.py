import warnings

import torch
from torch import nn


class Partitions(nn.Module):
    def __init__(self, bounds):
        super().__init__()

        lower, upper = bounds

        self.register_buffer('lower', lower)
        self.register_buffer('upper', upper)

    @property
    def volume(self):
        return self.width.prod(dim=-1).sum()

    @property
    def width(self):
        return self.upper - self.lower

    def sample(self, num_samples):
        perm = torch.randperm(self.lower.size(0))
        idx = perm[:num_samples]

        return Partitions(
            (self.lower[idx], self.upper[idx])
        )


class Partitioning(nn.Module):
    def __init__(self, initial, safe, unsafe, state_space=None):
        super().__init__()

        self.initial = Partitions(initial)
        self.safe = Partitions(safe)
        self.unsafe = Partitions(unsafe)
        self.state_space = Partitions(state_space) if state_space is not None else None
