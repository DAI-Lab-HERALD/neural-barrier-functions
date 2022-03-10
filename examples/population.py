import logging
import log

import os.path
from argparse import ArgumentParser

import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from model import PopulationBarrier
from dynamics import Population
from partitioning import population_partitioning

from learned_cbf.barrier import NeuralSBF


logger = logging.getLogger(__name__)


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
    logger.info(f'loss: [{loss_barrier:>7f}/{unsafety_prob:>7f}/{loss:>7f}], gamma: {gamma:>7f}, beta: {beta:>7f}, kappa: {kappa:>4f}')


def train(sbf, args):
    # dataset = PartitioningSubsampleDataset(population_partitioning(), batch_size=2000, iter_per_epoch=100)
    # dataloader = DataLoader(dataset, batch_size=None, num_workers=8)

    optimizer = optim.Adam(sbf.parameters(), lr=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    kappa = 1.0

    for epoch in trange(200, desc='Epoch', colour='red', position=0, leave=False):
        if epoch >= 190:
            kappa = 0.0

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

    logger.info(f'[{method.upper()}] certified: {certified}, prob unsafety: {unsafety_prob:>7f}, gamma: {gamma:>7f}, beta: {beta:>7f}')


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
    parser.add_argument('--save-path', type=str, default='models/sbf.pth', help='Path to save SBF to.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
