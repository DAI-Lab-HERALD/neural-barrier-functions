import copy
import logging

import os.path

import torch
from bound_propagation import BoundModelFactory, HyperRectangle
from torch import optim, nn
from torch.distributions import Normal
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import trange, tqdm

from .dynamics import DubinsFixedStrategy, BoundDubinsFixedStrategy, \
    DubinsCarNoActuation, DubinsCarStrategyComposition, DubinsCarNNStrategy, \
    BoundDubinSelect, DubinSelect, BoundDubinsCarNominalUpdate, DubinsCarNominalUpdate
from .partitioning import dubins_car_partitioning
from .plot import plot_bounds_2d

from neural_barrier_functions.certifier import SplittingNeuralSBFCertifier, AdditiveGaussianSplittingNeuralSBFCertifier
from neural_barrier_functions.learner import AdversarialNeuralSBF, EmpiricalNeuralSBF
from neural_barrier_functions.networks import FCNNBarrierNetwork
from neural_barrier_functions.dataset import StochasticSystemDataset
from neural_barrier_functions.discretization import ButcherTableau, BoundButcherTableau
from neural_barrier_functions.bounds import NBFBoundModelFactory
from neural_barrier_functions.monte_carlo import monte_carlo_simulation

logger = logging.getLogger(__name__)


def step(robust_learner, empirical_learner, optimizer, partitioning, kappa, epoch, empirical_only):
    optimizer.zero_grad(set_to_none=True)
    robust_learner.dynamics.resample()

    if empirical_only:
        loss = empirical_learner.loss(partitioning, kappa)
    else:
        loss = robust_learner.loss(partitioning, kappa, method='ibp')
        # loss = 0.5 * empirical_learner.loss(partitioning, kappa) + 0.5 * robust_learner.loss(partitioning, kappa, method='crown_ibp_interval')

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
            # plot_partitioning(partitioning, config['dynamics']['safe_set'])

            partitioning = partitioning.to(args.device)
            step(robust_learner, empirical_learner, optimizer, partitioning, kappa, epoch, config['training']['empirical_only'])

        if epoch % config['training']['test_every'] == config['training']['test_every'] - 1:
            test(certifier, config['test'], kappa)
            save(robust_learner, args, epoch)

        scheduler.step()
        kappa *= 0.97

    while not certifier.certify(method='crown_interval', batch_size=config['test']['ibp_batch_size']):
        logger.info(f'Current violation: {certifier.barrier_violation(method="crown_interval", batch_size=config["test"]["ibp_batch_size"])[0]}')
        for partitioning in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
            # plot_partitioning(partitioning, config['dynamics']['safe_set'])

            partitioning = partitioning.to(args.device)
            step(robust_learner, empirical_learner, optimizer, partitioning, 0.0, config['training']['epochs'], config['training']['empirical_only'])

    logger.info('Training complete')


def reinforcement_learning(strategy, dynamics, args, config):
    assert isinstance(strategy, DubinsCarNNStrategy)

    q_network1 = nn.Sequential(
        nn.Linear(4, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 1)
    ).to(args.device)

    # factory = BoundModelFactory()
    # q_network1 = factory.build(q_network1)

    q_network2 = nn.Sequential(
        nn.Linear(4, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 1)
    ).to(args.device)

    q_network1_target = copy.deepcopy(q_network1)
    q_network2_target = copy.deepcopy(q_network2)

    optimA = optim.Adam(strategy.parameters(), lr=1e-3)
    optimC1 = optim.Adam(q_network1.parameters(), lr=1e-3)
    optimC2 = optim.Adam(q_network2.parameters(), lr=1e-3)

    replay_memory_size = 10000
    batch_size = 250
    gamma = 0.99

    states = []
    actions = []
    rewards = []
    next_states = []
    masks = []

    for epoch in trange(2000):
        state = dynamics.sample_initial(1).to(args.device)

        for iter in range(40):
            with torch.no_grad():
                action = Normal(strategy(state), 0.01).sample().clamp(min=-1.0, max=1.0)
                next_state = dynamics[1:](torch.cat([state, action], dim=-1))
                next_state = next_state[torch.randint(next_state.size(0), (1,))][0].detach()
                reward = 8 * dynamics.goal(next_state).float() - 2 * dynamics.unsafe(next_state).float() \
                         - (~dynamics.state_space(next_state)).float() - (2.0 - next_state[..., 1]) + 2.0
                done = (dynamics.goal(next_state) | dynamics.unsafe(next_state) | ~dynamics.state_space(next_state)).item()

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                masks.append(1 - done)

                state = next_state

                if done:
                    break

        print('Epoch: {}, Iteration: {}, ({}/{}/{})'.format(epoch, iter, dynamics.goal(next_state).item(), dynamics.unsafe(next_state).item(), not dynamics.state_space(next_state).item()))

        samples = torch.randint(len(states), (batch_size,))
        batch_states = torch.cat(states)[samples]
        batch_actions = torch.cat(actions)[samples]
        batch_rewards = torch.cat(rewards)[samples]
        batch_next_states = torch.cat(next_states)[samples]
        batch_masks = torch.tensor(masks)[samples].to(args.device)

        with torch.no_grad():
            next_action = Normal(strategy(batch_next_states), 0.01).sample().clamp(min=-1.0, max=1.0)
            next_q_value = torch.min(
                q_network1_target(torch.cat([batch_next_states, next_action], dim=-1)),
                q_network2_target(torch.cat([batch_next_states, next_action], dim=-1))
            )
            q_target = batch_rewards.unsqueeze(-1) + batch_masks.unsqueeze(-1) * gamma * next_q_value

        critic1_loss = F.mse_loss(q_network1(torch.cat([batch_states, batch_actions], dim=-1)), q_target)
        critic2_loss = F.mse_loss(q_network2(torch.cat([batch_states, batch_actions], dim=-1)), q_target)

        optimC1.zero_grad()
        optimC2.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        optimC1.step()
        optimC2.step()

        action_dist = Normal(strategy(batch_states), 0.01)
        action = action_dist.rsample().clamp(min=-1.0, max=1.0)
        q_value = torch.min(q_network1(torch.cat([batch_states, action], dim=-1)), q_network2(torch.cat([batch_states, action], dim=-1)))
        actor_loss = (0.2 * action_dist.log_prob(action) - q_value).mean()

        optimA.zero_grad()
        actor_loss.backward()
        optimA.step()

        states = states[-replay_memory_size:]
        actions = actions[-replay_memory_size:]
        rewards = rewards[-replay_memory_size:]
        next_states = next_states[-replay_memory_size:]
        masks = masks[-replay_memory_size:]

        soft_update(q_network1_target, q_network1, 0.005)
        soft_update(q_network2_target, q_network2, 0.005)

    save(strategy, args, 'rl-final')


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def save(model, args, state):
    folder = os.path.dirname(args.save_path)
    os.makedirs(folder, exist_ok=True)

    path = args.save_path.format(state=state)
    torch.save(model.state_dict(), path)


def load(model, args, state):
    path = args.save_path.format(state=state)
    model.load_state_dict(torch.load(path))


def build_strategy(config):
    if config['strategy'] == 'no_actuation':
        return DubinsCarNoActuation()
    elif config['strategy'] == 'neural_network':
        return DubinsCarNNStrategy()
    elif config['strategy'] == 'fossil':
        return DubinsFixedStrategy()
    else:
        raise ValueError('Strategy not supported')


def dubins_car_main(args, config):
    logger.info('Constructing model')
    strategy = build_strategy(config)
    dynamics = DubinsCarStrategyComposition(config['dynamics'], strategy).to(args.device)

    if config['experiment_type'] == 'barrier_function':
        factory = NBFBoundModelFactory()
        factory.register(DubinsCarNominalUpdate, BoundDubinsCarNominalUpdate)
        factory.register(DubinsFixedStrategy, BoundDubinsFixedStrategy)
        factory.register(DubinSelect, BoundDubinSelect)
        factory.register(ButcherTableau, BoundButcherTableau)

        barrier = FCNNBarrierNetwork(network_config=config['model']).to(args.device)
        initial_partitioning = dubins_car_partitioning(config, dynamics).to(args.device)
        robust_learner = AdversarialNeuralSBF(barrier, dynamics, factory, horizon=config['dynamics']['horizon']).to(args.device)
        empirical_learner = EmpiricalNeuralSBF(barrier, dynamics, horizon=config['dynamics']['horizon']).to(args.device)

        if args.task in ['test', 'plot']:
            load(robust_learner, args, 'final')

        if args.task == 'train':
            certifier = SplittingNeuralSBFCertifier(barrier, dynamics, factory, initial_partitioning, horizon=config['dynamics']['horizon'], max_set_size=20).to(args.device)

            if isinstance(strategy, DubinsCarNNStrategy):
                load(strategy, args, 'rl-final')

            train(robust_learner, empirical_learner, certifier, args, config)
            save(robust_learner, args, 'final')
        elif args.task == 'test':
            certifier = AdditiveGaussianSplittingNeuralSBFCertifier(barrier, dynamics, factory, initial_partitioning, horizon=config['dynamics']['horizon']).to(args.device)
            test(certifier, config['test'])
        elif args.task == 'plot':
            plot_bounds_2d(barrier, dynamics, args, config)
    elif config['experiment_type'] == 'rl':
        reinforcement_learning(strategy, dynamics, args, config)
    elif config['experiment_type'] == 'monte_carlo':
        monte_carlo_simulation(args, dynamics, config)
    else:
        raise ValueError('Invalid experiment type')
