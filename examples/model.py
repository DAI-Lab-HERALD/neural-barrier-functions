import math

from torch import nn

from learned_cbf.barrier import Barrier


class TanhLinear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        # nn.init.zeros_(self.bias)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class FinalLinear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize to uniform(0, 2) to avoid choking the learning with zero grad
        nn.init.uniform_(self.bias, 0.0, 2.0)


class PopulationBarrier(Barrier):
    def __init__(self, *args, num_hidden=128):
        if args:
            # To support __get_index__ of nn.Sequential when slice indexing
            # CROWN is doing this underlying
            super().__init__(*args)
        else:
            super().__init__(
                nn.Linear(2, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                FinalLinear(num_hidden, 1),
                nn.ReLU(),
            )
