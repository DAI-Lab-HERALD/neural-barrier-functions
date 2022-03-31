import numpy as np
import torch.distributions
from torch.utils.data import Dataset

from partitioning import Partitioning


class DubinDataset(Dataset):
    def __init__(self, config, dynamics, adjust_overlap=True):
        self.batch_size = config['batch_size']
        self.iter_per_epoch = config['iter_per_epoch']
        self.eps = torch.tensor(config['eps'])

        self.dynamics = dynamics
        self.adjust_overlap = adjust_overlap

        self.dist = torch.distributions.Uniform(torch.tensor([-2.0, -2.0, -np.pi / 2]), torch.tensor([2.0, 2.0, np.pi / 2]))

    def __getitem__(self, item):
        valid, partitioning = self.generate()

        while not valid:
            valid, partitioning = self.generate()

        return partitioning

    def generate(self):
        num_particles = self.batch_size // 4
        initial = self.dynamics.sample_initial(num_particles)
        safe = self.dynamics.sample_safe(num_particles)
        unsafe = self.dynamics.sample_unsafe(num_particles)
        state_space = self.dynamics.sample_state_space(num_particles)

        if self.adjust_overlap:
            overlap_initial = self.dynamics.initial(safe, self.eps) & ~self.dynamics.initial(safe)
            if torch.any(overlap_initial):
                initial = torch.cat([initial, safe[overlap_initial]], dim=0)

            overlap_safe = self.dynamics.safe(unsafe, self.eps)
            overlap_unsafe = self.dynamics.unsafe(safe, self.eps)

            if torch.any(overlap_safe) and torch.any(overlap_unsafe):
                safe, unsafe = torch.cat([safe, unsafe[overlap_safe]], dim=0), torch.cat([unsafe, safe[overlap_unsafe]], dim=0)
            elif torch.any(overlap_safe):
                safe = torch.cat([safe, unsafe[overlap_safe]], dim=0)
            elif torch.any(overlap_unsafe):
                unsafe = torch.cat([unsafe, safe[overlap_unsafe]], dim=0)

        partitioning = Partitioning(
            (initial - self.eps, initial + self.eps),
            (safe - self.eps, safe + self.eps),
            (unsafe - self.eps, unsafe + self.eps),
            (state_space - self.eps, state_space + self.eps)
        )
        return True, partitioning

        # x = self.dist.sample((self.batch_size,))
        # assert torch.all(self.dynamics.state_space(x))
        #
        # initial_mask = self.dynamics.initial(x)
        # safe_mask = self.dynamics.safe(x)
        # unsafe_mask = self.dynamics.unsafe(x)
        #
        # if torch.any(initial_mask) and torch.any(safe_mask) and torch.any(unsafe_mask):
        #     lower_x, upper_x = x - self.eps, x + self.eps
        #
        #     partitioning = Partitioning(
        #         (lower_x[initial_mask], upper_x[initial_mask]),
        #         (lower_x[safe_mask], upper_x[safe_mask]),
        #         (lower_x[unsafe_mask], upper_x[unsafe_mask]),
        #         (lower_x, upper_x)
        #     )
        #
        #     return True, partitioning
        # else:
        #     return False, None

    def __len__(self):
        return self.iter_per_epoch

