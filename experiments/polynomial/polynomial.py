import logging

import os.path

import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange, tqdm

from certifier import NeuralSBFCertifier
from learner import AdversarialNeuralSBF
from .dynamics import Polynomial
from .partitioning import polynomial_partitioning
from .plot import plot_bounds_2d

from learned_cbf.partitioning import PartitioningSubsampleDataset, PartitioningDataLoader
from learned_cbf.networks import FCNNBarrierNetwork

logger = logging.getLogger(__name__)


def step(learner, optimizer, partitioning, kappa, epoch):
    optimizer.zero_grad(set_to_none=True)
    loss = learner.loss(partitioning, kappa, method='optimal' if epoch >= 20 else 'ibp')
    loss.backward()
    optimizer.step()


@torch.no_grad()
def status_method(certifier, kappa, method, batch_size):
    loss_barrier = certifier.barrier_violation(method=method, batch_size=batch_size)
    unsafety_prob, beta, gamma = certifier.unsafety_prob(return_beta_gamma=True, method=method, batch_size=batch_size)

    loss_barrier, unsafety_prob = loss_barrier.item(), unsafety_prob.item()
    beta, gamma = beta.item(), gamma.item()
    logger.info(f'[{method.upper()}] loss: ({loss_barrier:>7f}/{unsafety_prob:>7f}), gamma: {gamma:>7f}, beta: {beta:>7f}, kappa: {kappa:>4f}')


@torch.no_grad()
def status(certifier, kappa, status_config):
    status_method(certifier, kappa, method='ibp', batch_size=status_config['ibp_batch_size'])
    status_method(certifier, kappa, method='crown_ibp_linear', batch_size=status_config['crown_ibp_batch_size'])
    status_method(certifier, kappa, method='optimal', batch_size=status_config['crown_ibp_batch_size'])


def train(learner, certifier, args, config):
    status(certifier, 1.0, config['training']['status'])

    dataset = PartitioningSubsampleDataset(polynomial_partitioning(config))
    dataloader = PartitioningDataLoader(dataset, batch_size=config['training']['batch_size'], drop_last=True)

    optimizer = optim.Adam(learner.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.97)
    kappa = 0.99

    for epoch in trange(200, desc='Epoch', colour='red', position=0, leave=False):
        for partitioning in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
            partitioning = partitioning.to(args.device)
            step(learner, optimizer, partitioning, kappa, epoch)

        if epoch % 10 == 9:
            status(certifier, kappa, config['training']['status'])

        scheduler.step()
        if epoch >= 10:
            kappa *= 0.99

    while not certifier.certify(method='ibp', batch_size=200):
        step(learner, optimizer, certifier.partitioning, 0.0, epoch)


@torch.no_grad()
def test_method(certifier, method, batch_size):
    certified = certifier.barrier_violation(method=method, batch_size=batch_size)
    unsafety_prob, beta, gamma = certifier.unsafety_prob(method=method, batch_size=batch_size, return_beta_gamma=True)
    unsafety_prob, beta, gamma = unsafety_prob.item(), beta.item(), gamma.item()

    logger.info(f'[{method.upper()}] certified: {certified}, prob unsafety: {unsafety_prob:>7f}, gamma: {gamma:>7f}, beta: {beta:>7f}')


@torch.no_grad()
def test(certifier, args, config):
    status_config = config['training']['status']

    test_method(certifier, method='ibp', batch_size=status_config['ibp_batch_size'])
    test_method(certifier, method='crown_ibp_linear', batch_size=status_config['crown_ibp_batch_size'])
    test_method(certifier, method='optimal', batch_size=status_config['crown_ibp_batch_size'])
    # test_method(sbf, method='crown_linear', batch_size=status_config['crown_ibp_batch_size'])


def save(learner, args):
    folder = os.path.dirname(args.save_path)
    os.makedirs(folder, exist_ok=True)

    torch.save(learner.state_dict(), args.save_path)


def polynomial_main(args, config):
    dynamics = Polynomial(config['dynamics']).to(args.device)
    barrier = FCNNBarrierNetwork(network_config=config['model']).to(args.device)
    partitioning = polynomial_partitioning(config).to(args.device)
    learner = AdversarialNeuralSBF(barrier, dynamics, horizon=config['dynamics']['horizon']).to(args.device)
    certifier = NeuralSBFCertifier(barrier, dynamics, partitioning, horizon=config['dynamics']['horizon']).to(args.device)

    # sbf.load_state_dict(torch.load(args.save_path))

    train(learner, certifier, args, config)
    save(learner, args)
    test(certifier, args, config)

    plot_bounds_2d(barrier, args)
