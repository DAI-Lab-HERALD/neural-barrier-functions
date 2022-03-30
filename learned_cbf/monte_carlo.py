import logging

import torch


logger = logging.getLogger(__name__)


def monte_carlo_simulation(dynamics, config):
    assert config['experiment_type'] == 'monte_carlo'

    num_particles = config['num_particles']
    horizon = config['dynamics']['horizon']

    x = dynamics.sample_initial(num_particles)
    traj = [x]

    unsafe = torch.full((num_particles,), False)
    out_of_bounds = torch.full((num_particles,), False)

    for _ in range(horizon):
        x = dynamics(x)
        # For each sample, pick one realization for next step
        particle_samples = torch.randint(x.size(0), (x.size(1),))
        x = x[particle_samples, torch.arange(x.size(1))]
        traj.append(x)

        out_of_bounds |= ~dynamics.state_space(x) & ~unsafe
        unsafe |= dynamics.unsafe(x) & ~out_of_bounds

    logger.info(f'Num particles: {num_particles}/unsafe: {unsafe.sum()}/out_of_bounds: {out_of_bounds.sum()}, ratio unsafe: {unsafe.sum() / num_particles}, ratio out_of_bounds: {out_of_bounds.sum() / num_particles}')

    return x, torch.stack(traj), unsafe, out_of_bounds
