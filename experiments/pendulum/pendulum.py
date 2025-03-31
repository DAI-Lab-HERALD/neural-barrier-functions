import logging

import os.path

import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from .partitioning import plot_partitioning, nndm_partitioning
from .plot import plot_bounds_2d, plot_contours, plot_heatmap, plot_nominal_dynamics
from .dynamics import NNDM

from neural_barrier_functions.certifier import SplittingNeuralSBFCertifier, AdditiveGaussianSplittingNeuralSBFCertifier
from neural_barrier_functions.learner import AdversarialNeuralSBF, EmpiricalNeuralSBF
from neural_barrier_functions.networks import FCNNBarrierNetwork
from neural_barrier_functions.dataset import StochasticSystemDataset
from neural_barrier_functions.bounds import NBFBoundModelFactory
from neural_barrier_functions.monte_carlo import monte_carlo_simulation


logger = logging.getLogger(__name__)


def step(robust_learner, empirical_learner, optimizer, partitioning, kappa, epoch, empirical_only):
    optimizer.zero_grad(set_to_none=True)
    robust_learner.dynamics.resample()

    if empirical_only:
        loss = empirical_learner.loss(partitioning, kappa)
    else:
        # Use IBP for the robust learner, since that's what SABR (state-of-the-art) is doing.
        # https://arxiv.org/abs/2210.04871
        # Caveat: They also do PGD to find the optimal point within the attack model, and use IBP over a small box
        # centered at this point.
        loss = robust_learner.loss(partitioning, kappa, method='ibp')

    loss.backward()
    torch.nn.utils.clip_grad_norm_(robust_learner.parameters(), 1.0)
    optimizer.step()


@torch.no_grad()
def test_method(certifier, method, batch_size, kappa=None):
    loss_barrier, ce = certifier.barrier_violation(method=method, batch_size=batch_size)
    unsafety_prob, beta, gamma = certifier.unsafety_prob(return_beta_gamma=True, method=method, batch_size=batch_size)

    loss_barrier, unsafety_prob = loss_barrier.item(), unsafety_prob.item()
    beta, gamma = beta.item(), gamma.item()
    msg = f'[{method.upper()}] certification: ({loss_barrier:>.10f}/{unsafety_prob:>7f}), gamma: {gamma:>7f}, beta: {beta:>7f}'
    if kappa is not None:
        msg += f', kappa: {kappa:>4f}'
    logger.info(msg)


@torch.no_grad()
def test(certifier, status_config, kappa=None, method='crown_linear'):
    test_method(certifier, method=method, batch_size=status_config['crown_ibp_batch_size'], kappa=kappa)


def train(robust_learner, empirical_learner, certifier, args, config):
    logger.info('Starting training')
    test(certifier, config['test'])

    dataset = StochasticSystemDataset(config['training'], robust_learner.dynamics)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    optimizer = optim.Adam(robust_learner.barrier.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.97)
    kappa = 1.0

    for epoch in trange(config['training']['epochs'], desc='Epoch', colour='red', position=0, leave=False):
        for partitioning in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
            # plot_partitioning(partitioning)

            partitioning = partitioning.to(args.device)
            step(robust_learner, empirical_learner, optimizer, partitioning, kappa, epoch, config['training']['empirical_only'])

        if epoch % config['training']['test_every'] == config['training']['test_every'] - 1:
            test(certifier, config['test'], kappa, 'ibp')
            save(robust_learner, args, epoch)

        scheduler.step()
        # Switch gradually from focusing on safety to focusing on violation
        kappa *= 0.97

    # Reduce learning rate massively to ensure that the last learning to ensure non-violation
    # does not ruin the safety certificate
    scheduler.step(config['training']['epochs'] * 2)

    # Since we are only computing state space and unsafe, we can scale more.
    certifier.max_set_size *= 10

    certified, violation, counterexample = certifier.certify(method='crown_linear', batch_size=config['test']['ibp_batch_size'])

    while not certified:
        logger.info(f'Current violation: {violation}')
        dataset.add_counterexample(counterexample)

        for partitioning in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
            # plot_partitioning(partitioning, config['dynamics']['safe_set'])

            partitioning = partitioning.to(args.device)
            step(robust_learner, empirical_learner, optimizer, partitioning, 0.0, config['training']['epochs'], config['training']['empirical_only'])

        certified, violation, counterexample = certifier.certify(method='crown_linear', batch_size=config['test']['ibp_batch_size'])

    logger.info('Training complete')


def save(learner, args, state):
    folder = os.path.dirname(args.save_path)
    os.makedirs(folder, exist_ok=True)

    path = args.save_path.format(state=state)

    torch.save(learner.state_dict(), path)


def load(model, args, state):
    path = args.save_path.format(state=state)
    model.load_state_dict(torch.load(path, map_location=args.device))


def pendulum_main(args, config):
    logger.info('Constructing model')
    dynamics = NNDM(config['dynamics']).to(device=args.device, dtype=torch.float64)

    if config['experiment_type'] == 'barrier_function':
        factory = NBFBoundModelFactory()
        barrier = FCNNBarrierNetwork(network_config=config['model']).to(args.device)
        initial_partitioning = nndm_partitioning(config, dynamics).to(args.device)
        robust_learner = AdversarialNeuralSBF(barrier, dynamics, factory, horizon=config['dynamics']['horizon']).to(args.device)
        empirical_learner = EmpiricalNeuralSBF(barrier, dynamics, horizon=config['dynamics']['horizon']).to(args.device)

        if args.task in ['test', 'plot']:
            load(robust_learner, args, 'final')

        if args.task == 'train':
            # Use SplittingNeuralSBFCertifier (based on samples) during training for speed
            certifier = SplittingNeuralSBFCertifier(barrier, dynamics, factory, initial_partitioning, horizon=config['dynamics']['horizon']).to(args.device)
            train(robust_learner, empirical_learner, certifier, args, config)
            save(robust_learner, args, 'final')
        elif args.task == 'test':
            # Use AdditiveGaussianSplittingNeuralSBFCertifier during testing for exact verification
            certifier = AdditiveGaussianSplittingNeuralSBFCertifier(barrier, dynamics, factory, initial_partitioning, horizon=config['dynamics']['horizon'], max_set_size=config['test']['max_set_size']).to(args.device)
            test(certifier, config['test'])
        elif args.task == 'plot':
            plot_contours(barrier, args, config)
            plot_bounds_2d(barrier, dynamics, args, config)
    elif config['experiment_type'] == 'monte_carlo':
        monte_carlo_simulation(args, dynamics, config)
    else:
        raise ValueError('Invalid experiment type')
