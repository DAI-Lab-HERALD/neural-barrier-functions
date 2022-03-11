import random
from typing import Optional, Sized, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataset import T_co
from torch._six import int_classes as _int_classes


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

        return Partitions((self.lower[idx], self.upper[idx]))

    def __getitem__(self, idx):
        return Partitions(
            (self.lower[idx], self.upper[idx])
        )

    def __len__(self):
        return self.lower.size(0)


class Partitioning(nn.Module):
    def __init__(self, initial, safe, unsafe, state_space=None):
        super().__init__()

        self.initial = self.convert(initial)
        self.safe = self.convert(safe)
        self.unsafe = self.convert(unsafe)
        self.state_space = self.convert(state_space)

    @staticmethod
    def convert(partitions):
        if partitions is None:
            return None
        elif isinstance(partitions, Partitions):
            return partitions
        else:
            return Partitions(partitions)

    def __len__(self):
        return len(self.initial) + \
               len(self.safe) + \
               len(self.unsafe) + \
               (len(self.state_space) if self.state_space else 0)

    def __getitem__(self, idx):
        initial_idx, safe_idx, unsafe_idx, state_space_idx = idx

        return Partitioning(
            self.initial[initial_idx],
            self.safe[safe_idx],
            self.unsafe[unsafe_idx],
            self.state_space[state_space_idx] if self.state_space else None,
        )


class PartitioningSubsampleDataset(Dataset):
    def __init__(self, base: Partitioning):
        self.base = base

    def __getitem__(self, idx) -> T_co:
        return self.base[idx]

    def __len__(self):
        return len(self.base)


class PartitioningBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: PartitioningSubsampleDataset, batch_size: int, drop_last=False, generator=None):
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        super().__init__(dataset)
        min_size = 4 if dataset.base.state_space else 3
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or batch_size <= min_size:
            raise ValueError('batch_size should be a positive integer value greater than or equal to {}, '
                             'but got batch_size={}'.format(min_size, batch_size))

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        # Set ratios -- round down, adjust later
        num_partitions = len(self.dataset.base)
        num_chunks = (num_partitions + self.batch_size - 1) // self.batch_size

        initial_batch_size = int((len(self.dataset.base.initial) / num_partitions) * self.batch_size)
        initial_batch_sizes = torch.full((num_chunks,), initial_batch_size)
        initial_batch_sizes[-1] += len(self.dataset.base.initial) - initial_batch_sizes.sum()

        safe_batch_size = int((len(self.dataset.base.safe) / num_partitions) * self.batch_size)
        safe_batch_sizes = torch.full((num_chunks,), safe_batch_size)
        safe_batch_sizes[-1] += len(self.dataset.base.safe) - safe_batch_sizes.sum()

        unsafe_batch_size = int((len(self.dataset.base.unsafe) / num_partitions) * self.batch_size)
        unsafe_batch_sizes = torch.full((num_chunks,), unsafe_batch_size)
        unsafe_batch_sizes[-1] += len(self.dataset.base.unsafe) - unsafe_batch_sizes.sum()

        if self.dataset.base.state_space:
            state_space_size = len(self.dataset.base.state_space)
            state_space_batch_size = int((state_space_size / num_partitions) * self.batch_size)
        else:
            state_space_size = 0
            state_space_batch_size = 0
        state_space_batch_sizes = torch.full((num_chunks,), state_space_batch_size)
        state_space_batch_sizes[-1] += state_space_size - state_space_batch_sizes.sum()

        # Adjust batch sizes to match self.batch_size on all but the last
        batch_sizes = initial_batch_sizes + safe_batch_sizes + unsafe_batch_sizes + state_space_batch_sizes
        next_lst = [0] * initial_batch_sizes[-1] + [1] * safe_batch_sizes[-1] + [2] * unsafe_batch_sizes[-1] + [3] * state_space_batch_sizes[-1]
        random.shuffle(next_lst)

        for i in range(num_chunks - 1):
            missing = self.batch_size - batch_sizes[i]
            for _ in range(missing):
                next_inc = next_lst.pop()
                if next_inc == 0:
                    initial_batch_sizes[i] += 1
                    initial_batch_sizes[-1] -= 1
                elif next_inc == 1:
                    safe_batch_sizes[i] += 1
                    safe_batch_sizes[-1] -= 1
                elif next_inc == 2:
                    unsafe_batch_sizes[i] += 1
                    unsafe_batch_sizes[-1] -= 1
                elif next_inc == 3:
                    state_space_batch_sizes[i] += 1
                    state_space_batch_sizes[-1] -= 1

                assert next_inc <= 4, 'next should never be 4 and there should always be some excess in the last batch'

        assert initial_batch_sizes[-1] >= 0
        assert safe_batch_sizes[-1] >= 0
        assert unsafe_batch_sizes[-1] >= 0
        assert state_space_batch_sizes[-1] >= 0

        # Generate sets of indices in shuffled
        initial_idx = torch.randperm(len(self.dataset.base.initial), generator=generator).split(initial_batch_sizes.tolist())
        safe_idx = torch.randperm(len(self.dataset.base.safe), generator=generator).split(safe_batch_sizes.tolist())
        unsafe_idx = torch.randperm(len(self.dataset.base.unsafe), generator=generator).split(unsafe_batch_sizes.tolist())
        if self.dataset.base.state_space:
            state_space_idx = torch.randperm(len(self.dataset.base.state_space), generator=generator) \
                .split(state_space_batch_sizes.tolist())
        else:
            state_space_idx = None

        # Check that the number of batches match
        assert len(initial_idx) == len(safe_idx)
        assert len(safe_idx) == len(unsafe_idx)
        assert state_space_idx is None or len(unsafe_idx) == len(state_space_idx)

        for i in range(len(self)):
            if self.dataset.base.state_space:
                yield initial_idx[i], safe_idx[i], unsafe_idx[i], state_space_idx[i]
            else:
                yield initial_idx[i], safe_idx[i], unsafe_idx[i], None

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class PartitioningDataLoader(DataLoader):
    def __init__(self, dataset: PartitioningSubsampleDataset, *args, batch_size: Optional[int] = 1, **kwargs):
        sampler = PartitioningBatchSampler(dataset, batch_size=batch_size)
        super().__init__(dataset, *args, sampler=sampler, batch_size=None, **kwargs)
