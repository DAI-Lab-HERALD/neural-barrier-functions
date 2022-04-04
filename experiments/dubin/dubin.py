import logging

import os.path

import torch
from bound_propagation import BoundModelFactory
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from monte_carlo import monte_carlo_simulation
from .dynamics import DubinsCarUpdate, BoundDubinsCarUpdate, DubinsFixedStrategy, BoundDubinsFixedStrategy, \
    DubinsCarNoActuation, BoundDubinsCarNoActuation, DubinsCarStrategyComposition, DubinsCarNNStrategy
from .partitioning import dubins_car_partitioning
from .plot import plot_bounds_2d

from learned_cbf.certifier import NeuralSBFCertifier
from learned_cbf.learner import AdversarialNeuralSBF, EmpiricalNeuralSBF
from learned_cbf.networks import FCNNBarrierNetwork
from learned_cbf.dataset import StochasticSystemDataset
from learned_cbf.discretization import Euler, BoundEuler, BoundRK4, RK4

logger = logging.getLogger(__name__)


def step(learner, optimizer, partitioning, kappa, epoch):
    optimizer.zero_grad(set_to_none=True)

    if isinstance(learner, EmpiricalNeuralSBF):
        loss = learner.loss(partitioning, kappa)
    else:
        loss = learner.loss(partitioning, kappa, method='combined', violation_normalization_factor=1.0)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
    optimizer.step()


@torch.no_grad()
def test_method(certifier, method, batch_size, kappa=None):
    loss_barrier = certifier.barrier_violation(method=method, batch_size=batch_size)
    unsafety_prob, beta, gamma = certifier.unsafety_prob(return_beta_gamma=True, method=method, batch_size=batch_size)

    loss_barrier, unsafety_prob = loss_barrier.item(), unsafety_prob.item()
    beta, gamma = beta.item(), gamma.item()
    msg = f'[{method.upper()}] certification: ({loss_barrier:>.10f}/{unsafety_prob:>7f}), gamma: {gamma:>7f}, beta: {beta:>7f}'
    if kappa is not None:
        msg += f', kappa: {kappa:>4f}'
    logger.info(msg)


@torch.no_grad()
def test(certifier, test_config, kappa=None):
    test_method(certifier, method='ibp', batch_size=test_config['ibp_batch_size'], kappa=kappa)
    # test_method(certifier, method='crown_ibp_linear', batch_size=test_config['crown_ibp_batch_size'], kappa=kappa)
    test_method(certifier, method='optimal', batch_size=test_config['crown_ibp_batch_size'], kappa=kappa)


def train(learner, certifier, args, config):
    logger.info('Starting training')
    test(certifier, config['test'])

    dataset = StochasticSystemDataset(config['training'], learner.dynamics)
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
        kappa *= 0.95

    while not certifier.certify(method='ibp', batch_size=config['test']['ibp_batch_size']):
        logger.info(f'Current violation: {certifier.barrier_violation(method="ibp", batch_size=config["test"]["ibp_batch_size"])}')
        for partitioning in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
            # plot_partitioning(partitioning, config['dynamics']['safe_set'])

            partitioning = partitioning.to(args.device)
            step(learner, optimizer, partitioning, 0.0, config['training']['epochs'])

    logger.info('Training complete')


def save(learner, args):
    folder = os.path.dirname(args.save_path)
    os.makedirs(folder, exist_ok=True)

    torch.save(learner.state_dict(), args.save_path)


def dubins_car_main(args, config):
    logger.info('Constructing model')
    dynamics = DubinsCarStrategyComposition(config['dynamics'], DubinsCarNNStrategy()).to(args.device)

    if config['experiment_type'] == 'barrier_function':
        factory = BoundModelFactory()
        factory.register(DubinsCarUpdate, BoundDubinsCarUpdate)
        factory.register(DubinsFixedStrategy, BoundDubinsFixedStrategy)
        factory.register(DubinsCarNoActuation, BoundDubinsCarNoActuation)
        factory.register(RK4, BoundRK4)

        barrier = FCNNBarrierNetwork(network_config=config['model']).to(args.device)
        partitioning = dubins_car_partitioning(config, dynamics).to(args.device)
        learner = AdversarialNeuralSBF(barrier, dynamics, factory, horizon=config['dynamics']['horizon']).to(args.device)
        certifier = NeuralSBFCertifier(barrier, dynamics, factory, partitioning, horizon=config['dynamics']['horizon']).to(args.device)

        # learner.load_state_dict(torch.load(args.save_path))

        train(learner, certifier, args, config)
        save(learner, args)
        test(certifier, config['test'])

        # plot_bounds_2d(barrier, dynamics, args, config)
    elif config['experiment_type'] == 'monte_carlo':
        monte_carlo_simulation(args, dynamics, config)
    else:
        raise ValueError('Invalid experiment type')
