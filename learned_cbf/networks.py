import math

from bound_propagation import crown_ibp, crown, ibp
from torch import nn

from bounds import bounds


class BarrierNetwork(nn.Sequential):
    def bounds(self, partitions, prefix=None, bound_lower=True, bound_upper=True, method='ibp', batch_size=None, **kwargs):
        if prefix is None:
            model = self
        else:
            model = nn.Sequential(*list(prefix.children()), *list(self.children()))

        return bounds(model, partitions, bound_lower=bound_lower, bound_upper=bound_upper, method=method, batch_size=batch_size, **kwargs)


@ibp
@crown
@crown_ibp
class FCNNBarrierNetwork(BarrierNetwork):
    activation_class_mapping = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh
    }

    def __init__(self, *args, network_config=None):
        if args:
            # To support __get_index__ of nn.Sequential when slice indexing
            # CROWN is doing this underlying
            super().__init__(*args)
        else:
            assert network_config is not None
            assert network_config['type'] == 'fcnn'

            activation_class = self.activation_class_mapping[network_config['activation']]

            # First hidden layer (since it requires another input dimension
            layers = [nn.Linear(network_config['input_dim'], network_config['hidden_nodes']), activation_class()]

            # Other hidden layers
            for _ in range(network_config['hidden_layers'] - 1):
                layers.append(nn.Linear(network_config['hidden_nodes'], network_config['hidden_nodes']))
                layers.append(activation_class())

            # Output layer (no activation)
            layers.append(nn.Linear(network_config['hidden_nodes'], 1))

            super().__init__(*layers)
