from abc import ABC, abstractmethod


class StochasticDynamics(ABC):
    @abstractmethod
    def sample(self, num_samples):
        raise NotImplementedError()
