import torch
from torch import nn, Tensor
import onnx2torch

from neural_barrier_functions.dynamics import AdditiveGaussianDynamics
from torch.distributions import Normal


class NNDM(AdditiveGaussianDynamics):
    def __init__(self, dynamics_config):
        # Load model from ONNX
        onnx_model_path = dynamics_config['nn_model']
        graph_model = onnx2torch.convert(onnx_model_path)

        # Assume that the model is structured as a Sequential
        modules = graph_model.children()
        nominal = nn.Sequential(*modules)
        super().__init__(nominal, **dynamics_config)

        self.safe_set = torch.as_tensor(dynamics_config['safe_set'][0]), torch.as_tensor(dynamics_config['safe_set'][1])
        self.initial_set = torch.as_tensor(dynamics_config['initial_set'][0]), torch.as_tensor(dynamics_config['initial_set'][1])
        # Let the state space be [2 * Xs_lower, 2 * Xs_upper]
        self._state_space = 1.2 * self.safe_set[0], 1.2 * self.safe_set[1]

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
