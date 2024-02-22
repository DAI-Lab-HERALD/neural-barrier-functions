import torch
from torch import nn, Tensor
import onnx2torch

from neural_barrier_functions.dynamics import AdditiveGaussianDynamics
from torch.distributions import Normal


class AdditiveNoise(nn.Linear):
    def __init__(self, num_samples, sigma):
        super().__init__(2, 2)

        self.num_samples = num_samples
        self.sigma = sigma

        del self.weight
        del self.bias

        self.register_buffer('weight', torch.as_tensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ]).unsqueeze(0), persistent=True)

        dist = Normal(torch.zeros((2,)), self.sigma)
        z = dist.sample((self.num_samples,))
        self.register_buffer('bias', z.unsqueeze(1), persistent=True)

    def resample(self):
        dist = Normal(torch.zeros((2,)), self.sigma)
        z = dist.sample((self.num_samples,))
        self.bias = z.unsqueeze(1).to(self.bias.device)

    def forward(self, input: Tensor) -> Tensor:
        self.resample()
        return input + self.bias


class NNDM(nn.Sequential, AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        self.sigma = torch.as_tensor(dynamics_config['sigma'])
        AdditiveGaussianDynamics.__init__(self, dynamics_config['num_samples'])

        # Load model from ONNX
        onnx_model_path = dynamics_config['nn_model']
        graph_model = onnx2torch.convert(onnx_model_path)

        # Assume that the model is structured as a Sequential
        modules = graph_model.children()
        nn.Sequential.__init__(self,
                               nn.Sequential(*modules),  # Nominal system
                               AdditiveNoise(self.num_samples, self.sigma))

        self.safe_set = torch.as_tensor(dynamics_config['safe_set'][0]), torch.as_tensor(dynamics_config['safe_set'][1])
        self.initial_set = torch.as_tensor(dynamics_config['initial_set'][0]), torch.as_tensor(dynamics_config['initial_set'][1])
        # Let the state space be [2 * Xs_lower, 2 * Xs_upper]
        self._state_space = 1.2 * self.safe_set[0], 1.2 * self.safe_set[1]

    @property
    def v(self):
        return torch.zeros_like(self.sigma), self.sigma

    @property
    def nominal_system(self):
        return self[0]

    def resample(self):
        self[1].resample()

    def initial(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            outside1 = torch.any(upper_x < self.initial_set[0].to(x.device), dim=-1)
            outside2 = torch.any(lower_x > self.initial_set[1].to(x.device), dim=-1)

            return ~outside1 & ~outside2

        return torch.all(self.initial_set[0].to(x.device) < x, dim=-1) & torch.all(x < self.initial_set[1].to(x.device), dim=-1)

    def sample_initial(self, num_particles):
        dist = torch.distributions.Uniform(self.initial_set[0], self.initial_set[1])
        x = dist.sample((num_particles,))

        return x

    def safe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            outside1 = torch.any(upper_x < self.safe_set[0].to(x.device), dim=-1)
            outside2 = torch.any(lower_x > self.safe_set[1].to(x.device), dim=-1)

            return ~outside1 & ~outside2

        return torch.all(self.safe_set[0].to(x.device) < x, dim=-1) & torch.all(x < self.safe_set[1].to(x.device), dim=-1)

    def sample_safe(self, num_particles):
        dist = torch.distributions.Uniform(self.safe_set[0], self.safe_set[1])
        x = dist.sample((num_particles,))

        return x

    def unsafe(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            outside1 = torch.any(lower_x < self.safe_set[0].to(x.device), dim=-1)
            outside2 = torch.any(upper_x > self.safe_set[1].to(x.device), dim=-1)

            return outside1 | outside2

        return torch.any(x < self.safe_set[0].to(x.device), dim=-1) | torch.any(self.safe_set[1].to(x.device) < x, dim=-1)

    def sample_unsafe(self, num_particles):
        x = self.sample_state_space(num_particles * 10)
        x = x[self.unsafe(x)]

        return x[:num_particles]

    def state_space(self, x, eps=None):
        if eps is not None:
            lower_x, upper_x = x - eps, x + eps

            outside1 = torch.any(upper_x < self._state_space[0].to(x.device), dim=-1)
            outside2 = torch.any(lower_x > self._state_space[1].to(x.device), dim=-1)

            return ~outside1 & ~outside2

        return torch.all(self._state_space[0].to(x.device) < x, dim=-1) & torch.all(x < self._state_space[1].to(x.device), dim=-1)

    def sample_state_space(self, num_particles):
        dist = torch.distributions.Uniform(self._state_space[0], self._state_space[1])
        x = dist.sample((num_particles,))

        return x

    @property
    def volume(self):
        return (self._state_space[1] - self._state_space[0]).prod()
