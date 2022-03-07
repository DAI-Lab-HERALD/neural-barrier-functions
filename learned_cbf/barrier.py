import torch
from torch import nn


class NeuralSBF(nn.Module):
    def __init__(self, barrier, dynamics, partitioning, horizon, num_samples):
        super().__init__()

        self.barrier = barrier
        self.dynamics = dynamics

        # Assumptions:
        # 1. Initial set, unsafe set, and safe set are partitioned.
        # 2. State space is optionally partitioned. If not, it should end with ReLU.
        # 3. Partitions containing the boundary of the safe / unsafe set belong to both (to ensure correctness)
        # 4. Partitions are hyperrectangular and non-overlapping
        self.partitioning = partitioning

        self.horizon = horizon
        self.num_samples = num_samples

        # self.rho = nn.Parameter(torch.empty((1,)))
        self.mu = nn.Parameter(torch.empty((1,)))
        self.nu = nn.Parameter(torch.empty((1,)))

        # TODO: Sample from state space (place on partitioning) and assert output is 1-D

    @property
    def alpha(self):
        """
        Parameterize alpha by rho to allow constraint free optimization.
        TODO: Change parameterization to allow [1, infty]
        :return: alpha = 1
        """
        return 1.0  # + self.rho.exp()

    @property
    def beta(self):
        """
        Parameterize beta by mu to allow constraint free optimization
        TODO: Change parameterization to allow [0, 1)
        :return: beta in range (0, 1)
        """
        return self.mu.sigmoid()

    @property
    def gamma(self):
        """
        Parameterize gamma by nu to allow constraint free optimization
        TODO: Change parameterization to allow [0, 1)
        :return: gamma in range (0, 1)
        """
        return self.nu.sigmoid()

    def loss(self, safety_weight=0.01):
        return (1.0 - safety_weight) * self.loss_barrier + safety_weight * self.loss_safety_prob

    @property
    def loss_barrier(self):
        return self.loss_init + self.loss_unsafe + self.loss_state_space + self.loss_expectation

    @property
    def loss_init(self):
        """
        Ensure that B(x) <= gamma for all x in X_0.
        :return: Loss for initial set
        """
        assert self.partitioning.initial is not None

        _, upper = self.barrier.crown_interval(self.partitioning.initial, upper_only=True)
        violation = (upper - self.gamma).clamp(min=0).sum()
        return violation / self.partitioning.initial.volume

    @property
    def loss_unsafe(self):
        """
        Ensure that B(x) >= 1 for all x in X_u.
        :return: Loss for unsafe set
        """
        assert self.partitioning.unsafe is not None

        lower, _ = self.barrier.crown_interval(self.partitioning.unsafe, lower_only=True)
        violation = (1 - lower).clamp(min=0).sum()
        return violation / self.partitioning.unsafe.volume

    @property
    def loss_state_space(self):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        if self.partitioning.state_space is not None:
            lower, _ = self.barrier.crown_interval(self.partitioning.state_space, bound_upper=False)
            violation = -lower.clamp(max=0).sum()
            return violation / self.partitioning.state_space.volume
        else:
            # Assume that dynamics ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
            return 0

    @property
    def loss_expectation(self):
        """
        Ensure that B(F(x, sigma)) <= B(x) / alpha + beta for all x in X_s.
        :return: Loss for the expectation constraint
        """
        assert self.partitioning.safe is not None

        dynamics = self.dynamics.sample(self.num_samples)

        _, upper = self.barrier.crown_interval(self.partitioning.safe, dynamics, bound_lower=False)
        lower, _ = self.barrier.crown_interval(self.partitioning.safe, bound_upper=False)
        violation = (upper.sum(dim=0) - lower / self.alpha - self.beta).clamp(min=0).sum()

        return violation / self.partitioning.safe.volume

    @property
    def loss_safety_prob(self):
        """
        gamma + beta * horizon is an upper bound for probability of B(x) >= 1 in horizon steps.
        If alpha > 1.0, we can do better but gamma + beta * horizon remains an upper bound.
        :return: Upper bound for (1 - safety probability).
        """
        if self.alpha == 1.0:
            bx_violation_prob = self.gamma + self.beta * self.horizon
        elif self.alpha * self.beta / (1 - self.alpha) <= 1:
            bx_violation_prob = 1 - (1 - self.gamma) * self.beta ** self.horizon
        else:
            bx_violation_prob = self.gamma * self.alpha ** (-self.horizon) + \
                (1 - self.alpha ** (-self.horizon)) * self.alpha * self.beta / (self.alpha - 1)

        return bx_violation_prob

    @torch.no_grad()
    def certify(self):
        """
        Certify that a trained barrier network is a valid barrier using the barrier loss
        Allow a small violation to account for potential numerical errors.
        :return: true if the barrier network is a valid barrier
        """
        return self.loss_barrier.item() <= 1.0e-10
