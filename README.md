# Neural barrier functions
Neural networks as barrier functions for stochastic discrete-time systems trained and verified using bound propagation.

To train:

    python experiments/main.py --device=<cpu|gpu> --config-path=<config-path> --save-path=models/<model-name>.{state}.pth --task=train

To certify:

    python experiments/main.py --device=<cpu|gpu> --config-path=<config-path> --save-path=models/<model-name>.{state}.pth --task=test

To plot:

    python experiments/main.py --device=<cpu|gpu> --config-path=<config-path> --save-path=models/<model-name>.{state}.pth --task=plot