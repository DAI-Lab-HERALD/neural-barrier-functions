# Neural barrier functions
Neural networks as barrier functions for stochastic discrete-time systems trained and verified using bound propagation.

To train:

    python experiments/main.py --device=<cpu|gpu> --config-path=<config-path> --save-path=models/<model-name>.{state}.pth --task=train

To certify:

    python experiments/main.py --device=<cpu|gpu> --config-path=<config-path> --save-path=models/<model-name>.{state}.pth --task=test

To plot:

    python experiments/main.py --device=<cpu|gpu> --config-path=<config-path> --save-path=models/<model-name>.{state}.pth --task=plot


## Authors
- [Frederik Baymler Mathiesen](https://www.baymler.com) - PhD student @ TU Delft

## Funding and support
- TU Delft

## Copyright notice:
Technische Universiteit Delft hereby disclaims all copyright
interest in the program “neural-barrier-functions” 
(neural networks as barrier functions with bound propagation)
written by the Frederik Baymler Mathiesen. [Name
Dean], Dean of [Name Faculty]

© 2022, Frederik Baymler Mathiesen, HERALD Lab, TU Delft