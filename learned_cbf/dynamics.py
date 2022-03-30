import abc


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
    def unsafe(self, x, eps=None):
        return NotImplementedError()

    @abc.abstractmethod
    def state_space(self, x, eps=None):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def volume(self):
        raise NotImplementedError()
