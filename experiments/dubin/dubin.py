import copy
import logging

import os.path

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import trange, tqdm

from .dynamics import DubinsCarUpdate, BoundDubinsCarUpdate, DubinsFixedStrategy, BoundDubinsFixedStrategy, \
    DubinsCarNoActuation, DubinsCarStrategyComposition, DubinsCarNNStrategy, \
    BoundDubinSelect, DubinSelect
from .partitioning import dubins_car_partitioning
from .plot import plot_bounds_2d

from learned_cbf.certifier import NeuralSBFCertifier, SplittingNeuralSBFCertifier
from learned_cbf.learner import AdversarialNeuralSBF, EmpiricalNeuralSBF
from learned_cbf.networks import FCNNBarrierNetwork
from learned_cbf.dataset import StochasticSystemDataset
from learned_cbf.discretization import ButcherTableau, BoundButcherTableau
from learned_cbf.bounds import LearnedCBFBoundModelFactory
from learned_cbf.monte_carlo import monte_carlo_simulation

logger = logging.getLogger(__name__)


def step(learner, optimizer, partitioning, kappa, epoch):
    optimizer.zero_grad(set_to_none=True)

    if isinstance(learner, EmpiricalNeuralSBF):
        loss = learner.loss(partitioning, kappa)
    else:
        loss = learner.loss(partitioning, kappa, method='crown_ibp_interval', violation_normalization_factor=1.0)

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
    # test_method(certifier, method='ibp', batch_size=test_config['ibp_batch_size'], kappa=kappa)
    # test_method(certifier, method='crown_ibp_interval', batch_size=test_config['crown_ibp_batch_size'], kappa=kappa)
    test_method(certifier, method='optimal', batch_size=test_config['crown_ibp_batch_size'], kappa=kappa)


def train(learner, certifier, args, config):
    logger.info('Starting training')
    test(certifier, config['test'])

    dataset = StochasticSystemDataset(config['training'], learner.dynamics)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=8)

    empirical_learner = EmpiricalNeuralSBF(learner.barrier, learner.dynamics, learner.horizon)

    optimizer = optim.Adam(learner.parameters(), lr=1e-3)
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
            save(learner, args, epoch)

        scheduler.step()
        kappa *= 0.97

    while not certifier.certify(method='ibp', batch_size=config['test']['ibp_batch_size']):
        logger.info(f'Current violation: {certifier.barrier_violation(method="ibp", batch_size=config["test"]["ibp_batch_size"])}')
        for partitioning in tqdm(dataloader, desc='Iteration', colour='red', position=1, leave=False):
            # plot_partitioning(partitioning, config['dynamics']['safe_set'])

            partitioning = partitioning.to(args.device)
            step(learner, optimizer, partitioning, 0.0, config['training']['epochs'])

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
    )

    q_network2 = nn.Sequential(
        nn.Linear(4, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 1)
    )

    q_network1_target = copy.deepcopy(q_network1)
    q_network2_target = copy.deepcopy(q_network2)

    strategy_target = copy.deepcopy(strategy)

    optimA = optim.Adam(strategy.parameters(), lr=1e-4)
    optimC1 = optim.Adam(q_network1.parameters(), lr=1e-4)
    optimC2 = optim.Adam(q_network2.parameters(), lr=1e-4)

    replay_memory_size = 10000
    batch_size = 200
    gamma = 0.97

    states = []
    actions = []
    rewards = []
    next_states = []
    masks = []

    for epoch in trange(10000):
        state = dynamics.sample_initial(1)

        for iter in range(40):
            with torch.no_grad():
                next_state = dynamics(state)
                next_state = next_state[torch.randint(next_state.size(0), (1,))][0].detach()
                reward = dynamics.safe(next_state).float() + 10 * dynamics.goal(next_state).float() \
                         - 10 * dynamics.unsafe(next_state).float() - 5 * (~dynamics.state_space(next_state)).float()
                done = (dynamics.goal(next_state) | dynamics.unsafe(next_state) | ~dynamics.state_space(next_state)).item()

                states.append(state)
                actions.append(strategy(state))
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
        batch_masks = torch.tensor(masks)[samples]

        with torch.no_grad():
            next_action = strategy_target(batch_next_states)
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

        action = strategy(batch_states)
        q_value = torch.min(q_network1(torch.cat([batch_states, action], dim=-1)), q_network2(torch.cat([batch_states, action], dim=-1)))
        actor_loss = -q_value.mean()

        optimA.zero_grad()
        actor_loss.backward()
        optimA.step()

        states = states[-replay_memory_size:]
        actions = actions[-replay_memory_size:]
        rewards = rewards[-replay_memory_size:]
        next_states = next_states[-replay_memory_size:]
        masks = masks[-replay_memory_size:]

        soft_update(q_network1_target, q_network1, 0.01)
        soft_update(q_network2_target, q_network2, 0.01)
        soft_update(strategy_target, strategy, 0.01)

    save(strategy, args, 'rl-final')


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def save(model, args, state):
    folder = os.path.dirname(args.save_path)
    os.makedirs(folder, exist_ok=True)

    path = args.save_path.format(state=state)

    torch.save(model.state_dict(), path)


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
        factory = LearnedCBFBoundModelFactory()
        factory.register(DubinsCarUpdate, BoundDubinsCarUpdate)
        factory.register(DubinsFixedStrategy, BoundDubinsFixedStrategy)
        factory.register(DubinSelect, BoundDubinSelect)
        factory.register(ButcherTableau, BoundButcherTableau)

        barrier = FCNNBarrierNetwork(network_config=config['model']).to(args.device)
        partitioning = dubins_car_partitioning(config, dynamics).to(args.device)
        learner = AdversarialNeuralSBF(barrier, dynamics, factory, horizon=config['dynamics']['horizon']).to(args.device)
        certifier = SplittingNeuralSBFCertifier(barrier, dynamics, factory, partitioning, horizon=config['dynamics']['horizon']).to(args.device)

        # learner.load_state_dict(torch.load(args.save_path))

        train(learner, certifier, args, config)
        save(learner, args, 'final')
        test(certifier, config['test'])

        # plot_bounds_2d(barrier, dynamics, args, config)
    elif config['experiment_type'] == 'rl':
        reinforcement_learning(strategy, dynamics, args, config)
    elif config['experiment_type'] == 'monte_carlo':
        monte_carlo_simulation(args, dynamics, config)
    else:
        raise ValueError('Invalid experiment type')
