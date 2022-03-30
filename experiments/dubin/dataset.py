import numpy as np
import torch.distributions
from torch.utils.data import Dataset

from partitioning import Partitioning


class DubinDataset(Dataset):
    def __init__(self, config, dynamics):
        self.batch_size = config['batch_size']
        self.iter_per_epoch = config['iter_per_epoch']
        self.eps = torch.tensor(config['eps'])

        self.dynamics = dynamics

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

