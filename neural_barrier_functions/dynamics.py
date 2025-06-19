import abc

import torch
from torch import distributions, nn


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


class AdditiveNoiseDynamics(StochasticDynamics, abc.ABC):
    pass


class AdditiveNoise(nn.Linear):
    def __init__(self, num_samples, mu, sigma):
        self.n = mu.size(0)
        super().__init__(in_features=self.n, out_features=self.n)

        self.num_samples = num_samples
        self.mu = mu
        self.sigma = sigma

        del self.weight
        del self.bias

        self.register_buffer('weight', torch.eye(self.n).unsqueeze(0))

        dist = distributions.Normal(torch.zeros((self.n,)), self.sigma)
        z = dist.sample((self.num_samples,))
        self.register_buffer('bias', z.unsqueeze(1))

    @property
    def v(self):
        return self.mu, self.sigma

    def resample(self):
        dist = distributions.Normal(self.mu, self.sigma)
        z = dist.sample((self.num_samples,))
        self.bias = z.unsqueeze(1).to(self.bias.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.resample()
        return input + self.bias


class AdditiveGaussianDynamics(nn.Sequential, AdditiveNoiseDynamics):
    def __init__(self, nominal_system, num_samples, mu, sigma, **kwargs):
        additive_noise = AdditiveNoise(
            num_samples, 
            torch.as_tensor(mu), 
            torch.as_tensor(sigma)
        )
        
        nn.Sequential.__init__(self, nominal_system, additive_noise)

    @property
    def nominal_system(self):
        return self[0]

    @property
    def v(self):
        return self[1].v

    def resample(self):
        self[1].resample()

    def prob_v(self, rect):
        loc, scale = self.v

        assert torch.all(scale >= 0.0)

        zero = torch.any((rect[0][..., scale == 0.0] > loc[scale == 0.0]) | (rect[1][..., scale == 0.0] < loc[scale == 0.0]), dim=-1)

        rect = (rect[0][..., scale > 0.0], rect[1][..., scale > 0.0])
        loc = loc[scale > 0]
        scale = scale[scale > 0]

        v = distributions.Normal(loc, scale)

        cdf_upper = v.cdf(rect[1])
        cdf_lower = v.cdf(rect[0])

        prob = (cdf_upper - cdf_lower).prod(dim=-1, keepdim=True)
        prob[zero] = 0.0

        return prob.clamp(min=0)
