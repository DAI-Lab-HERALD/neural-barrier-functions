from argparse import ArgumentParser

import torch

from dubin.dubin import dubins_car_main
from pendulum.pendulum import pendulum_main
from linear.linear import linear_main
from polynomial_4d.polynomial_4d import polynomial_4d_main
from polynomial.polynomial import polynomial_main
from log import configure_logging
from population.population import population_main
from neural_barrier_functions.utils import load_config


def main(args):
    config = load_config(args.config_path)

    if config['system'] == 'population':
        population_main(args, config)
    elif config['system'] == 'linear':
        linear_main(args, config)
    elif config['system'] == 'pendulum':
        pendulum_main(args, config)
    elif config['system'] == 'polynomial':
        polynomial_main(args, config)
    elif config['system'] == 'polynomial_4d':
        polynomial_4d_main(args, config)
    elif config['system'] == 'dubin':
        dubins_car_main(args, config)
    else:
        raise ValueError(f'System {config["system"]} not defined')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cpu',
                        help='Select device for tensor operations.')
    parser.add_argument('--config-path', type=str, help='Path to configuration of experiment.')
    parser.add_argument('--save-path', type=str, default='models/sbf.pth', help='Path to save SBF to.')
    parser.add_argument('--log-file', type=str, help='Path to log file.')
    parser.add_argument('--task', type=str, choices=['train', 'test', 'plot'], default='train', help='Train will learn a barrier and save to file. Test and plot with load the barrier and do their respective operations.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args.log_file)
    torch.set_default_dtype(torch.float64)
    main(args)
