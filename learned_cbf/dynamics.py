import abc
from abc import abstractmethod

from torch import nn


class StochasticDynamics(nn.Sequential, abc.ABC):
    def __init__(self, *args, num_samples=None):
        super().__init__(*args)

        assert num_samples is not None
        self.num_samples = num_samples

    @abstractmethod
    def safe(self, x):
        raise NotImplementedError()

    @abstractmethod
    def initial(self, x):
        raise NotImplementedError()

    @abstractmethod
    def state_space(self, x):
        raise NotImplementedError()

    def unsafe(self, x):
        return ~self.safe(x)
