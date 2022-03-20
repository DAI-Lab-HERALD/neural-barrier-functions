import abc
from abc import abstractmethod

from torch import nn


class StochasticDynamics(nn.Sequential, abc.ABC):
    def __init__(self, *args, num_samples=None):
        super().__init__(*args)

        assert num_samples is not None
        self.num_samples = num_samples

    @abstractmethod
    def initial(self, x, eps=None):
        raise NotImplementedError()

    @abstractmethod
    def safe(self, x, eps=None):
        raise NotImplementedError()

    @abstractmethod
    def unsafe(self, x, eps=None):
        return NotImplementedError()

    @abstractmethod
    def state_space(self, x, eps=None):
        raise NotImplementedError()

    @property
    @abstractmethod
    def volume(self):
        raise NotImplementedError()
