import logging

import os.path

import torch
from bound_propagation import BoundModelFactory
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from partitioning import PartitioningSubsampleDataset, PartitioningDataLoader
from .dataset import PolynomialDataset
from .dynamics import Polynomial, BoundPolynomial
from .partitioning import polynomial_partitioning
from .plot import plot_bounds_2d

from learned_cbf.certifier import NeuralSBFCertifier
from learned_cbf.learner import AdversarialNeuralSBF, EmpiricalNeuralSBF
from learned_cbf.networks import FCNNBarrierNetwork, ResidualBarrierNetwork

logger = logging.getLogger(__name__)


def step(learner, optimizer, partitioning, kappa, epoch):
    optimizer.zero_grad(set_to_none=True)

    if isinstance(learner, EmpiricalNeuralSBF):
        loss = learner.loss(partitioning.state_space.center, kappa)
    else:
        loss = learner.loss(partitioning, kappa, method='ibp', violation_normalization_factor=100.0)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
    optimizer.step()


@torch.no_grad()
def test_method(certifier, method, batch_size, kappa=None):
    loss_barrier = certifier.barrier_violation(method=method, batch_size=batch_size)
    unsafety_prob, beta, gamma = certifier.unsafety_prob(return_beta_gamma=True, method=method, batch_size=batch_size)

    loss_barrier, unsafety_prob = loss_barrier.item(), unsafety_prob.item()
    beta, gamma = beta.item(), gamma.item()
    msg = f'[{method.upper()}] certification: ({loss_barrier:>11f}/{unsafety_prob:>7f}), gamma: {gamma:>7f}, beta: {beta:>7f}'
    if kappa is not None:
        msg += f', kappa: {kappa:>4f}'
    logger.info(msg)


@torch.no_grad()
def test(certifier, status_config, kappa=None):
    test_method(certifier, method='ibp', batch_size=status_config['ibp_batch_size'], kappa=kappa)
    # test_method(certifier, method='crown_ibp_linear', batch_size=status_config['crown_ibp_batch_size'], kappa=kappa)
    # test_method(certifier, method='optimal', batch_size=status_config['crown_ibp_batch_size'], kappa=kappa)


def train(learner, certifier, args, config):
    logger.info('Starting training')
    test(certifier, config['test'])

    dataset = PolynomialDataset(config['training'], learner.dynamics)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=8)

    empirical_learner = EmpiricalNeuralSBF(learner.barrier, learner.dynamics, learner.horizon)

    optimizer = optim.AdamW(learner.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.97)
    kappa = 1.0

    for epoch in trange(config['training']['epochs'], desc='Epoch', colour='red', position=0, leave=False):
        for partitioning in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
            # plot_partitioning(partitioning, config['dynamics']['safe_set'])

            partitioning = partitioning.to(args.device)

            if epoch < config['training']['empirical_epochs']:
                step(empirical_learner, optimizer, partitioning, kappa, epoch)
            else:
                step(learner, optimizer, partitioning, kappa, epoch)

        if epoch % config['training']['test_every'] == config['training']['test_every'] - 1:
            test(certifier, config['test'], kappa)

        scheduler.step()
        kappa *= 0.97

    # while not certifier.certify(method='optimal', batch_size=config['test']['ibp_batch_size']):
    #     logger.info(f'Current violation: {certifier.barrier_violation(method="optimal", batch_size=config["test"]["ibp_batch_size"])}')
    #     for partitioning in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
    #         # plot_partitioning(partitioning, config['dynamics']['safe_set'])
    #
    #         partitioning = partitioning.to(args.device)
    #         step(learner, optimizer, partitioning, 0.0, config['training']['epochs'])

    logger.info('Training complete')


def subsample_partitioning(partitioning, config):
    quarter_batch = config['training']['iter_per_epoch'] // 4
    initial_idx = torch.randperm(len(partitioning.initial))[:quarter_batch]
    safe_idx = torch.randperm(len(partitioning.safe))[:quarter_batch]
    unsafe_idx = torch.randperm(len(partitioning.unsafe))[:quarter_batch]
    state_space_idx = torch.randperm(len(partitioning.state_space))[:quarter_batch]
    idx = initial_idx, safe_idx, unsafe_idx, state_space_idx

    return partitioning[idx]


def save(learner, args):
    folder = os.path.dirname(args.save_path)
    os.makedirs(folder, exist_ok=True)

    torch.save(learner.state_dict(), args.save_path)


def polynomial_main(args, config):
    logger.info('Constructing model')

    factory = BoundModelFactory()
    factory.register(Polynomial, BoundPolynomial)
    dynamics = Polynomial(config['dynamics']).to(args.device)
    barrier = FCNNBarrierNetwork(network_config=config['model']).to(args.device)
    partitioning = polynomial_partitioning(config, dynamics).to(args.device)
    learner = AdversarialNeuralSBF(barrier, dynamics, factory, horizon=config['dynamics']['horizon']).to(args.device)
    certifier = NeuralSBFCertifier(barrier, dynamics, factory, partitioning, horizon=config['dynamics']['horizon']).to(args.device)

    # learner.load_state_dict(torch.load(args.save_path))

    train(learner, certifier, args, config)
    save(learner, args)
    test(certifier, config['test'])

    plot_bounds_2d(barrier, dynamics, args, config)