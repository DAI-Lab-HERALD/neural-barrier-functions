import math
import os.path
from argparse import ArgumentParser

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Normal
from torch.nn import init
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from learned_cbf.barrier import NeuralSBF, Barrier
from learned_cbf.dynamics import StochasticDynamics
from learned_cbf.partitioning import Partitioning, PartitioningSubsampleDataset


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


def plot_partitioning(partitioning):
    fig, ax = plt.subplots()

    for lower, width in zip(partitioning.initial.lower, partitioning.initial.width):
        rect = plt.Rectangle(lower, width[0], width[1], color='g', alpha=0.1, linewidth=1)
        ax.add_patch(rect)

    for lower, width in zip(partitioning.safe.lower, partitioning.safe.width):
        rect = plt.Rectangle(lower, width[0], width[1], color='b', alpha=0.1, linewidth=1)
        ax.add_patch(rect)

    for lower, width in zip(partitioning.unsafe.lower, partitioning.unsafe.width):
        rect = plt.Rectangle(lower, width[0], width[1], color='r', alpha=0.1, linewidth=1)
        ax.add_patch(rect)

    circle_init = plt.Circle((0, 0), 1.0, color='g', fill=False)
    ax.add_patch(circle_init)

    circle_safe = plt.Circle((0, 0), 2.0, color='r', fill=False)
    ax.add_patch(circle_safe)

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()


def population_partitioning():
    x1_space = torch.linspace(-3.0, 3.0, 41)
    x2_space = torch.linspace(-3.0, 3.0, 41)

    cell_width = torch.stack([(x1_space[1] - x1_space[0]) / 2, (x2_space[1] - x2_space[0]) / 2])
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)
    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    closest_point = torch.min(lower_x.abs(), upper_x.abs())
    farthest_point = torch.max(lower_x.abs(), upper_x.abs())

    initial_mask = closest_point[:, 0]**2 + closest_point[:, 1]**2 <= 1.0**2
    safe_mask = closest_point[:, 0]**2 + closest_point[:, 1]**2 <= 2.0**2
    unsafe_mask = farthest_point[:, 0]**2 + farthest_point[:, 1]**2 >= 2.0**2

    partitioning = Partitioning(
        (lower_x[initial_mask], upper_x[initial_mask]),
        (lower_x[safe_mask], upper_x[safe_mask]),
        (lower_x[unsafe_mask], upper_x[unsafe_mask]),
        (lower_x, upper_x)
    )

    # plot_partitioning(partitioning)

    return partitioning


class TanhLinear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        # nn.init.zeros_(self.bias)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


class FinalLinear(nn.Linear):
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize to 1 to avoid choking the learning with zero grad
        nn.init.constant_(self.bias, 0.0)


class PopulationBarrier(Barrier):
    def __init__(self, *args, num_hidden=128):
        if args:
            # To support __get_index__ of nn.Sequential when slice indexing
            # CROWN is doing this underlying
            super().__init__(*args)
        else:
            super().__init__(
                nn.Linear(2, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, 1),
            )


def step(optimizer, sbf, kappa):
    optimizer.zero_grad(set_to_none=True)
    loss = sbf.loss(kappa)
    loss.backward()
    optimizer.step()


@torch.no_grad()
def status(sbf, kappa):
    loss_barrier = sbf.loss_barrier()
    unsafety_prob, beta, gamma = sbf.unsafety_prob(return_beta_gamma=True)
    loss = sbf.loss(kappa)

    loss_barrier, unsafety_prob, loss = loss_barrier.item(), unsafety_prob.item(), loss.item()
    beta, gamma = beta.item(), gamma.item()
    print(f"loss: [{loss_barrier:>7f}/{unsafety_prob:>7f}/{loss:>7f}], gamma: {gamma:>7f}, beta: {beta:>7f}, kappa: {kappa:>4f}")


def train(sbf, args):
    # dataset = PartitioningSubsampleDataset(population_partitioning(), batch_size=2000, iter_per_epoch=100)
    # dataloader = DataLoader(dataset, batch_size=None, num_workers=8)

    optimizer = optim.Adam(sbf.parameters(), lr=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    kappa = 1.0

    for epoch in trange(200, desc='Epoch', colour='red', position=0, leave=False):
        for iteration in trange(100, desc='Iteration', colour='red', position=1, leave=False):
            step(optimizer, sbf, kappa)

        status(sbf, kappa)
        scheduler.step()
        kappa *= 0.99


@torch.no_grad()
def test_method(sbf, args, method):
    certified = sbf.certify(method=method, batch_size=20)
    unsafety_prob, beta, gamma = sbf.unsafety_prob(method=method, batch_size=20, return_beta_gamma=True)
    unsafety_prob, beta, gamma = unsafety_prob.item(), beta.item(), gamma.item()

    print(method)
    print(f'certified: {certified}, prob unsafety: {unsafety_prob:>7f}, gamma: {gamma:>7f}, beta: {beta:>7f}')


@torch.no_grad()
def test(sbf, args):
    test_method(sbf, args, 'ibp')
    test_method(sbf, args, 'crown_ibp')
    test_method(sbf, args, 'crown')


def save(sbf, args):
    folder = os.path.dirname(args.save_path)
    os.makedirs(folder, exist_ok=True)

    torch.save(sbf.state_dict(), args.save_path)


def main(args):
    barrier = PopulationBarrier().to(args.device)
    dynamics = Population(num_samples=500).to(args.device)
    partitioning = population_partitioning().to(args.device)
    sbf = NeuralSBF(barrier, dynamics, partitioning, horizon=10).to(args.device)

    train(sbf, args)
    test(sbf, args)
    save(sbf, args)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cuda',
                        help='Select device for tensor operations.')
    parser.add_argument('--dim', choices=[1, 2], type=int, default=1, help='Dimensionality of the noisy sine.')
    parser.add_argument('--save-path', type=str, default='models/sbf.pth', help='Path to save SBF to.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
