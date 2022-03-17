from argparse import ArgumentParser

import torch

from log import configure_logging
from population.population import population_main
from utils import load_config


def main(args):
    config = load_config(args.config_path)

    if config['system'] == 'population':
        population_main(args, config)
    elif config['system'] == 'polynomial':
        polynomial_main(args, config)
    else:
        raise ValueError(f'System {config["system"]} not defined')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cuda',
                        help='Select device for tensor operations.')
    parser.add_argument('--config-path', type=str, help='Path to configuration of experiment.')
    parser.add_argument('--save-path', type=str, default='models/sbf.pth', help='Path to save SBF to.')
    parser.add_argument('--log-file', type=str, help='Path to log file.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args.log_file)
    main(args)
