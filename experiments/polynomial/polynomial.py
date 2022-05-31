import logging

import os.path

import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from .dynamics import Polynomial, PolynomialUpdate, BoundPolynomialUpdate, NominalPolynomialUpdate, \
    BoundNominalPolynomialUpdate
from .partitioning import polynomial_partitioning, plot_partitioning
from .plot import plot_bounds_2d

from learned_cbf.certifier import NeuralSBFCertifier, SplittingNeuralSBFCertifier, \
    AdditiveGaussianSplittingNeuralSBFCertifier
from learned_cbf.learner import AdversarialNeuralSBF, EmpiricalNeuralSBF
from learned_cbf.networks import FCNNBarrierNetwork
from learned_cbf.discretization import ButcherTableau, BoundButcherTableau
from learned_cbf.bounds import LearnedCBFBoundModelFactory
from learned_cbf.dataset import StochasticSystemDataset
from learned_cbf.monte_carlo import monte_carlo_simulation


logger = logging.getLogger(__name__)


def step(robust_learner, empirical_learner, optimizer, partitioning, kappa, epoch, empirical_only):
    optimizer.zero_grad(set_to_none=True)

    if empirical_only:
        loss = empirical_learner.loss(partitioning, kappa)
    else:
        loss = robust_learner.loss(partitioning, kappa, method='crown_interval')
        # loss = 0.5 * empirical_learner.loss(partitioning, kappa) + 0.5 * robust_learner.loss(partitioning, kappa, method='crown_ibp_interval')

    loss.backward()
    torch.nn.utils.clip_grad_norm_(robust_learner.parameters(), 1.0)
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
    # test_method(certifier, method='ibp', batch_size=test_config['ibp_batch_size'], kappa=kappa)
    test_method(certifier, method='crown_interval', batch_size=test_config['crown_ibp_batch_size'], kappa=kappa)
    # test_method(certifier, method='optimal', batch_size=test_config['crown_ibp_batch_size'], kappa=kappa)


def train(robust_learner, empirical_learner, certifier, args, config):
    logger.info('Starting training')
    test(certifier, config['test'])

    dataset = StochasticSystemDataset(config['training'], robust_learner.dynamics)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=8)

    optimizer = optim.Adam(robust_learner.barrier.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.97)
    kappa = 1.0

    for epoch in trange(config['training']['epochs'], desc='Epoch', colour='red', position=0, leave=False):
        for partitioning in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
            # plot_partitioning(partitioning)

            partitioning = partitioning.to(args.device)
            step(robust_learner, empirical_learner, optimizer, partitioning, kappa, epoch, config['training']['empirical_only'])

        if epoch % config['training']['test_every'] == config['training']['test_every'] - 1:
            test(certifier, config['test'], kappa)
            save(robust_learner, args, epoch)

        scheduler.step()
        kappa *= 0.97

    while not certifier.certify(method='crown_interval', batch_size=config['test']['ibp_batch_size']):
        logger.info(f'Current violation: {certifier.barrier_violation(method="crown_interval", batch_size=config["test"]["ibp_batch_size"])}')
        for partitioning in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
            # plot_partitioning(partitioning)

            partitioning = partitioning.to(args.device)
            step(robust_learner, empirical_learner, optimizer, partitioning, 0.0, config['training']['epochs'], config['training']['empirical_only'])

    logger.info('Training complete')


def save(learner, args, state):
    folder = os.path.dirname(args.save_path)
    os.makedirs(folder, exist_ok=True)

    path = args.save_path.format(state=state)

    torch.save(learner.state_dict(), path)


def load(model, args, state):
    path = args.save_path.format(state=state)
    model.load_state_dict(torch.load(path, map_location=args.device))


def polynomial_main(args, config):
    logger.info('Constructing model')
    dynamics = Polynomial(config['dynamics']).to(args.device)

    if config['experiment_type'] == 'barrier_function':
        factory = LearnedCBFBoundModelFactory()
        factory.register(PolynomialUpdate, BoundPolynomialUpdate)
        factory.register(NominalPolynomialUpdate, BoundNominalPolynomialUpdate)
        factory.register(ButcherTableau, BoundButcherTableau)

        barrier = FCNNBarrierNetwork(network_config=config['model']).to(args.device)
        initial_partitioning = polynomial_partitioning(config, dynamics).to(args.device)
        robust_learner = AdversarialNeuralSBF(barrier, dynamics, factory, horizon=config['dynamics']['horizon']).to(args.device)
        empirical_learner = EmpiricalNeuralSBF(barrier, dynamics, horizon=config['dynamics']['horizon']).to(args.device)

        if args.task in ['test', 'plot']:
            load(robust_learner, args, 'final')

        if args.task == 'train':
            certifier = SplittingNeuralSBFCertifier(barrier, dynamics, factory, initial_partitioning, horizon=config['dynamics']['horizon']).to(args.device)
            train(robust_learner, empirical_learner, certifier, args, config)
            save(robust_learner, args, 'final')
        elif args.task == 'test':
            certifier = AdditiveGaussianSplittingNeuralSBFCertifier(barrier, dynamics, factory, initial_partitioning, horizon=config['dynamics']['horizon']).to(args.device)
            test(certifier, config['test'])
        elif args.task == 'plot':
            plot_bounds_2d(barrier, dynamics, args, config)
    elif config['experiment_type'] == 'monte_carlo':
        monte_carlo_simulation(args, dynamics, config)
    else:
        raise ValueError('Invalid experiment type')
