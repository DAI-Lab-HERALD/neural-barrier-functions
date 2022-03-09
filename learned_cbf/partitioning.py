import warnings

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


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
        perm = torch.randperm(len(self))
        idx = perm[:num_samples]

        return self.lower[idx], self.upper[idx]

    def __getitem__(self, idx):
        return Partitions(
            (self.lower[idx], self.upper[idx])
        )

    def __len__(self):
        return self.lower.size(0)


class Partitioning(nn.Module):
    def __init__(self, initial, safe, unsafe, state_space=None):
        super().__init__()

        self.initial = Partitions(initial)
        self.safe = Partitions(safe)
        self.unsafe = Partitions(unsafe)
        self.state_space = Partitions(state_space) if state_space is not None else None


class PartitioningSubsampleDataset(Dataset):
    def __init__(self, base: Partitioning, batch_size: int, iter_per_epoch: int):
        self.base = base
        self.batch_size = batch_size
        self.iter_per_epoch = iter_per_epoch

    def __getitem__(self, index) -> T_co:
        return Partitioning(
            self.base.initial.sample(self.batch_size),
            self.base.safe.sample(self.batch_size),
            self.base.unsafe.sample(self.batch_size),
            self.base.state_space.sample(self.batch_size) if self.base.state_space is not None else None
        )

    def __len__(self):
        return self.iter_per_epoch
