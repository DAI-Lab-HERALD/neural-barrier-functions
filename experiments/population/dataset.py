import torch.distributions
from torch.utils.data import Dataset

from partitioning import Partitioning


class PopulationDataset(Dataset):
    def __init__(self, config, dynamics):
        self.batch_size = config['batch_size']
        self.iter_per_epoch = config['iter_per_epoch']
        self.eps = config['eps']

        self.dynamics = dynamics

        if self.dynamics.safe_set_type == 'circle':
            self.dist = torch.distributions.Uniform(torch.tensor([-3.0, -3.0]), torch.tensor([3.0, 3.0]))
        elif self.dynamics.safe_set_type == 'annulus':
            self.dist = torch.distributions.Uniform(torch.tensor([0.0, 0.0]), torch.tensor([4.5, 4.5]))
        else:
            raise ValueError('Invalid safe set for population')

    def __getitem__(self, item):
        valid, partitioning = self.generate()

        while not valid:
            valid, partitioning = self.generate()

        return partitioning

    def generate(self):
        x = self.dist.sample((self.batch_size,))
        assert torch.all(self.dynamics.state_space(x))

        initial_mask = self.dynamics.initial(x, self.eps)
        safe_mask = self.dynamics.initial(x, self.eps)
        unsafe_mask = self.dynamics.initial(x, self.eps)

        if torch.any(initial_mask) and  torch.any(safe_mask) and torch.any(unsafe_mask):
            lower_x, upper_x = x - self.eps, x + self.eps

            partitioning = Partitioning(
                (lower_x[initial_mask], upper_x[initial_mask]),
                (lower_x[safe_mask], upper_x[safe_mask]),
                (lower_x[unsafe_mask], upper_x[unsafe_mask]),
                (lower_x, upper_x)
            )

            return True, partitioning
        else:
            return False, None

    def __len__(self):
        return self.iter_per_epoch

