import math
from argparse import ArgumentParser
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Normal
from torch.nn import init
from tqdm import trange

from learned_cbf.barrier import NeuralSBF, Barrier
from learned_cbf.dynamics import StochasticDynamics
from learned_cbf.partitioning import Partitioning


class PopulationStep(nn.Linear):
    # x[1] = juveniles
    # x[2] = adults

    sigma = 0.1
    fertility_rate = 0.2
    survival_juvenile = 0.3
    survival_adult = 0.8

    def __init__(self, num_samples):
        super().__init__(2, 2)

        del self.weight
        del self.bias

        self.register_buffer('weight', torch.as_tensor([
            [0.0, self.fertility_rate],
            [self.survival_juvenile, self.survival_adult]
        ]), persistent=True)

        dist = Normal(0.0, self.sigma)
        z = dist.sample((num_samples,))
        self.register_buffer('bias', torch.stack([torch.zeros_like(z), z], dim=-1), persistent=True)


class Population(StochasticDynamics):
    def __init__(self, num_samples):
        super().__init__(
            PopulationStep(num_samples),
            num_samples=num_samples
        )

        x1_space = torch.linspace(-2.0, 2.0, 11)
        x2_space = torch.linspace(-2.0, 2.0, 11)

        cell_width = torch.stack([(x1_space[1] - x1_space[0]) / 2, (x2_space[1] - x2_space[0]) / 2])
        x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2
        x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

        cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)
        lower_x, upper_x = cell_centers - cell_width, + cell_centers + cell_width

        closest_point = torch.min(lower_x.abs(), upper_x.abs())
        farthest_point = torch.max(lower_x.abs(), upper_x.abs())

        initial_mask = closest_point[:, 0]**2 + closest_point[:, 1]**2 <= 1.0**2
        safe_mask = closest_point[:, 0]**2 + closest_point[:, 1]**2 <= 2.0**2
        unsafe_mask = farthest_point[:, 0]**2 + farthest_point[:, 1]**2 >= 2.0**2

        self.partitioning = Partitioning(
            (lower_x[initial_mask], upper_x[initial_mask]),
            (lower_x[safe_mask], upper_x[safe_mask]),
            (lower_x[unsafe_mask], upper_x[unsafe_mask]),
            (lower_x, upper_x)
        )


class MyLinear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


class PopulationBarrier(Barrier):
    def __init__(self, *args, num_hidden=16):
        if args:
            # To support __get_index__ of nn.Sequential when slice indexing
            # CROWN (and implicitly CROWN-IBP) is doing this underlying
            super().__init__(*args)
        else:
            super().__init__(
                MyLinear(2, num_hidden),
                nn.Tanh(),
                MyLinear(num_hidden, num_hidden),
                nn.Tanh(),
                nn.Linear(num_hidden, 1)
            )


def train(sbf):
    optimizer = optim.AdamW(sbf.parameters())

    for iteration in trange(10000):
        loss = sbf.loss()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            with torch.no_grad():
                loss_barrier = sbf.loss_barrier()
                loss_safety_prob = sbf.loss_safety_prob()
                loss = sbf.loss()

            loss_barrier, loss_safety_prob, loss = loss_barrier.item(), loss_safety_prob.item(), loss.item()
            gamma, beta = sbf.gamma.item(), sbf.beta.item()
            print(f"loss: [{loss_barrier:>7f}/{loss_safety_prob:>7f}/{loss:>7f}], gamma: {gamma}, beta: {beta}")


def main(args):
    barrier = PopulationBarrier().to(args.device)
    dynamics = Population(num_samples=200).to(args.device)
    sbf = NeuralSBF(barrier, dynamics, dynamics.partitioning, horizon=10).to(args.device)

    train(sbf)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cuda',
                        help='Select device for tensor operations')
    parser.add_argument('--dim', choices=[1, 2], type=int, default=1, help='Dimensionality of the noisy sine')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
