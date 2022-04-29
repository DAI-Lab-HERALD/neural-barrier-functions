import abc

from torch import distributions


class StochasticDynamics(abc.ABC):
    def __init__(self, num_samples=None):
        assert num_samples is not None
        self.num_samples = num_samples

    @abc.abstractmethod
    def initial(self, x, eps=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_initial(self, num_particles):
        raise NotImplementedError()

    @abc.abstractmethod
    def safe(self, x, eps=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_safe(self, num_particles):
        raise NotImplementedError()

    @abc.abstractmethod
    def unsafe(self, x, eps=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_unsafe(self, num_particles):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_space(self, x, eps=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_state_space(self, num_particles):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def volume(self):
        raise NotImplementedError()


class AdditiveGaussianDynamics(StochasticDynamics, abc.ABC):
    @property
    @abc.abstractmethod
    def nominal_system(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def G(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def v(self):
        raise NotImplementedError()

    def prob_v(self, rect):
        v = self.v
        assert isinstance(v, distributions.Normal)

        return None
