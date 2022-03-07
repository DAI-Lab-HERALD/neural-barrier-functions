from torch import nn


class StochasticDynamics(nn.Sequential):
    def __init__(self, *args, num_samples=None):
        super().__init__(*args)

        assert num_samples is not None
        self.num_samples = num_samples
