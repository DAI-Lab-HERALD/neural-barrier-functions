import logging

import log

import os.path
from argparse import ArgumentParser

import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange, tqdm

from dynamics import Population
from networks import FCNNBarrierNetwork
from partitioning import population_partitioning

from learned_cbf.barrier import NeuralSBF
from learned_cbf.partitioning import PartitioningSubsampleDataset, PartitioningDataLoader
from utils import load_config

logger = logging.getLogger(__name__)


def step(optimizer, sbf, kappa):
    optimizer.zero_grad(set_to_none=True)
    loss = sbf.loss(kappa)
    loss.backward()
    optimizer.step()


@torch.no_grad()
def status_method(sbf, kappa, method, batch_size):
    loss_barrier = sbf.loss_barrier(method=method, batch_size=batch_size)
    unsafety_prob, beta, gamma = sbf.unsafety_prob(return_beta_gamma=True, method=method, batch_size=batch_size)
    loss = sbf.loss(kappa, method=method, batch_size=batch_size)

    loss_barrier, unsafety_prob, loss = loss_barrier.item(), unsafety_prob.item(), loss.item()
    beta, gamma = beta.item(), gamma.item()
    logger.info(f'[{method.upper()}] loss: ({loss_barrier:>7f}/{unsafety_prob:>7f}/{loss:>7f}), gamma: {gamma:>7f}, beta: {beta:>7f}, kappa: {kappa:>4f}')


@torch.no_grad()
def status(sbf, kappa, status_config):
    status_method(sbf, kappa, method='ibp', batch_size=status_config['ibp_batch_size'])
    status_method(sbf, kappa, method='crown_ibp_linear', batch_size=status_config['crown_ibp_batch_size'])


def train(sbf, args, config):
    full_partitioning = sbf.partitioning
    status(sbf, 1.0, config['training']['status'])

    dataset = PartitioningSubsampleDataset(population_partitioning(config['partitioning']))
    dataloader = PartitioningDataLoader(dataset, batch_size=config['training']['batch_size'], drop_last=True)

    optimizer = optim.AdamW(sbf.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.98)
    kappa = 0.99

    for epoch in trange(300, desc='Epoch', colour='red', position=0, leave=False):
        for subsample in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
            sbf.partitioning = subsample.to(args.device)
            step(optimizer, sbf, kappa)

        if epoch % 10 == 9:
            sbf.partitioning = full_partitioning
            status(sbf, kappa, config['training']['status'])

        scheduler.step()
        if epoch >= 10:
            kappa *= 0.99

    kappa = 0.0
    sbf.partitioning = full_partitioning
    while not sbf.certify(method='ibp', batch_size=200):
        step(optimizer, sbf, kappa)


@torch.no_grad()
def test_method(sbf, method, batch_size):
    certified = sbf.loss_barrier(method=method, batch_size=batch_size)
    unsafety_prob, beta, gamma = sbf.unsafety_prob(method=method, batch_size=batch_size, return_beta_gamma=True)
    unsafety_prob, beta, gamma = unsafety_prob.item(), beta.item(), gamma.item()

    logger.info(f'[{method.upper()}] certified: {certified}, prob unsafety: {unsafety_prob:>7f}, gamma: {gamma:>7f}, beta: {beta:>7f}')


@torch.no_grad()
def test(sbf, args, config):
    status_config = config['training']['status']

    test_method(sbf, method='ibp', batch_size=status_config['ibp_batch_size'])
    test_method(sbf, method='crown_ibp_linear', batch_size=status_config['crown_ibp_batch_size'])
    # test_method(sbf, 'crown_linear', batch_size=1)


def save(sbf, args):
    folder = os.path.dirname(args.save_path)
    os.makedirs(folder, exist_ok=True)

    torch.save(sbf.state_dict(), args.save_path)


def main(args):
    config = load_config(args.config_path)

    barrier = FCNNBarrierNetwork(network_config=config['model']).to(args.device)
    dynamics = Population(config['dynamics']).to(args.device)
    partitioning = population_partitioning(config['partitioning']).to(args.device)
    sbf = NeuralSBF(barrier, dynamics, partitioning, horizon=2).to(args.device)

    train(sbf, args, config)
    save(sbf, args)
    test(sbf, args, config)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cuda',
                        help='Select device for tensor operations.')
    parser.add_argument('--config-path', type=str, help='Path to configuration of experiment.')
    parser.add_argument('--save-path', type=str, default='models/sbf.pth', help='Path to save SBF to.')
    parser.add_argument('--log-file', type=str, help='Path to log file.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    log.configure_logging(args.log_file)
    main(args)
