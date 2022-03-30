import logging

import torch


logger = logging.getLogger(__name__)


def monte_carlo_simulation(dynamics, config):
    assert config['experiment_type'] == 'monte_carlo'

    num_particles = config['num_particles']
    horizon = config['dynamics']['horizon']

    x = dynamics.sample_initial(num_particles)

    unsafe = None
    out_of_bounds = None

    for _ in range(horizon):
        x = dynamics(x)
        # For each sample, pick one realization for next step
        x = x[torch.randint(0, x.size(0)), torch.arange(x.size(1))]

        if unsafe is None:
            unsafe = x[dynamics.unsafe(x)]
        else:
            unsafe = torch.cat([unsafe, x[dynamics.unsafe(x)]])

        if out_of_bounds is None:
            out_of_bounds = x[~dynamics.state_space(x)]
        else:
            out_of_bounds = torch.cat([out_of_bounds, x[~dynamics.state_space(x)]])

        x = x[dynamics.safe(x) & dynamics.state_space]

    logger.info(f'Safe: {x.size(0)}/unsafe: {unsafe.size(0)}/out_of_bounds: {out_of_bounds.size(0)}, ratio unsafe: {unsafe.size(0) / num_particles}, ratio out_of_bounds: {out_of_bounds.size(0) / num_particles}')

    return x, unsafe, out_of_bounds
