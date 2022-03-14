import torch
from torch import nn

from .partitioning import Partitions


class NeuralSBF(nn.Module):
    def __init__(self, barrier, dynamics, partitioning, horizon, train_beta=False, train_gamma=False):
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

        # self.rho = nn.Parameter(torch.empty((1,)))

        if train_beta:
            self.mu = nn.Parameter(torch.empty((1,)))
        else:
            self.register_buffer('mu', None)

        if train_gamma:
            self.nu = nn.Parameter(torch.empty((1,)))
        else:
            self.register_buffer('nu', None)

        # TODO: Sample from state space (place on partitioning) and assert output is 1-D

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.normal_(self.rho, 0, 0.1)

        if self.mu is not None:
            nn.init.normal_(self.mu, 0, 0.1)
        if self.nu is not None:
            nn.init.normal_(self.nu, 0, 0.1)

    @property
    def alpha(self):
        """
        Parameterize alpha by rho to allow constraint free optimization.
        TODO: Change parameterization to allow [1, infty]
        :return: alpha = 1
        """
        return 1.0  # + F.softplus(self.rho)

    def beta(self, **kwargs):
        """
        Parameterize beta by mu to allow constraint free optimization
        :return: beta in range [0, 1)
        """
        if self.mu is not None:
            # Only returns beta in range (0, 1)
            # TODO: Change parameterization to allow [0, 1)
            return self.mu.sigmoid()
        else:
            assert self.partitioning.safe is not None

            with torch.no_grad():
                _, upper = self.barrier.interval(self.partitioning.safe, self.dynamics, bound_lower=False, **kwargs)
                lower, _ = self.barrier.interval(self.partitioning.safe, bound_upper=False, **kwargs)

                expectation_no_beta = (upper.mean(dim=0) - lower / self.alpha).partition_max()
                idx = expectation_no_beta.argmax()

                # if expectation_no_beta[idx].item() <= 0.0:
                #     return torch.tensor(0.0, device=upper.device)

                beta_max_partition = self.partitioning.safe[idx]
                beta_max_partition = Partitions((
                    beta_max_partition.lower.unsqueeze(0),
                    beta_max_partition.upper.unsqueeze(0)
                ))

            _, upper = self.barrier.interval(beta_max_partition, self.dynamics, bound_lower=False, **kwargs)
            lower, _ = self.barrier.interval(beta_max_partition, bound_upper=False, **kwargs)
            beta = (upper.mean(dim=0) - lower / self.alpha).partition_max().clamp(min=0)

            return beta

    def gamma(self, **kwargs):
        """
        Parameterize gamma by nu to allow constraint free optimization
        :return: gamma in range [0, 1)
        """
        if self.nu is not None:
            # Only returns gamma in range (0, 1)
            # TODO: Change parameterization to allow [0, 1)
            return self.nu.sigmoid()
        else:
            assert self.partitioning.initial is not None

            with torch.no_grad():
                _, upper = self.barrier.interval(self.partitioning.initial, bound_lower=False, **kwargs)
                idx = upper.partition_max().argmax()

                # if upper[idx].item() <= 0.0:
                #     return torch.tensor(0.0, device=upper.device)

                gamma_max_partition = self.partitioning.initial[idx]
                gamma_max_partition = Partitions((
                    gamma_max_partition.lower.unsqueeze(0),
                    gamma_max_partition.upper.unsqueeze(0)
                ))

            _, upper = self.barrier.interval(gamma_max_partition, bound_lower=False, **kwargs)
            gamma = upper.partition_max().clamp(min=0)

            return gamma

    def loss(self, safety_weight=0.5, **kwargs):
        if safety_weight == 1.0:
            return self.loss_safety_prob(**kwargs)
        elif safety_weight == 0.0:
            return self.loss_barrier(**kwargs)

        loss_barrier = self.loss_barrier(**kwargs)
        loss_safety_prob = self.loss_safety_prob(**kwargs)

        # if loss_barrier.item() >= 1.0e-10:
        #     return loss_barrier
        # else:
        #     return loss_safety_prob

        return (1.0 - safety_weight) * loss_barrier + safety_weight * loss_safety_prob

    def loss_barrier(self, **kwargs):
        loss = self.loss_state_space(**kwargs) + self.loss_unsafe(**kwargs)
        if self.mu is not None:
            loss += self.loss_expectation(**kwargs)
        if self.nu is not None:
            loss += self.loss_init(**kwargs)

        return loss

    def loss_init(self, **kwargs):
        """
        Ensure that B(x) <= gamma for all x in X_0.
        :return: Loss for initial set
        """
        assert self.partitioning.initial is not None

        _, upper = self.barrier.interval(self.partitioning.initial, bound_upper=False, **kwargs)
        violation = (upper - self.gamma()).partition_max().clamp(min=0).sum()
        return violation / self.partitioning.initial.volume

    def loss_unsafe(self, **kwargs):
        """
        Ensure that B(x) >= 1 for all x in X_u.
        :return: Loss for unsafe set
        """
        assert self.partitioning.unsafe is not None

        lower, _ = self.barrier.interval(self.partitioning.unsafe, bound_upper=False, **kwargs)
        violation = (1 - lower).partition_min().clamp(min=0).sum()
        return violation / self.partitioning.unsafe.volume

    def loss_state_space(self, **kwargs):
        """
        Ensure that B(x) >= 0 for all x in X. If no partitioning is available,
        assume that barrier network ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
        :return: Loss for state space (zero if not partitioned)
        """
        if self.partitioning.state_space is not None:
            lower, _ = self.barrier.interval(self.partitioning.state_space, bound_upper=False, **kwargs)
            violation = (0 - lower).partition_min().clamp(min=0).sum()
            return violation / self.partitioning.state_space.volume
        else:
            # Assume that dynamics ends with ReLU, i.e. B(x) >= 0 for all x in R^n.
            return 0.0

    def loss_expectation(self, **kwargs):
        """
        Ensure that B(F(x, sigma)) <= B(x) / alpha + beta for all x in X_s.
        :return: Loss for the expectation constraint
        """
        assert self.partitioning.safe is not None

        _, upper = self.barrier.interval(self.partitioning.safe, self.dynamics, bound_lower=False, **kwargs)
        lower, _ = self.barrier.interval(self.partitioning.safe, bound_upper=False, **kwargs)
        violation = (upper.mean(dim=0) - lower / self.alpha - self.beta()).partition_max().clamp(min=0).sum()

        return violation / self.partitioning.safe.volume

    def loss_safety_prob(self, **kwargs):
        """
        gamma + beta * horizon is an upper bound for probability of B(x) >= 1 in horizon steps.
        But we need to account for the fact that we backprop through dynamics in beta, num_samples times.
        :return: Upper bound for (1 - safety probability) adjusted to construct loss.
        """
        return self.gamma(**kwargs) + self.beta(**kwargs) * self.horizon

    @torch.no_grad()
    def unsafety_prob(self, return_beta_gamma=False, **kwargs):
        """
        gamma + beta * horizon is an upper bound for probability of B(x) >= 1 in horizon steps.
        If alpha > 1.0, we can do better but gamma + beta * horizon remains an upper bound.
        :return: Upper bound for (1 - safety probability).
        """
        beta = self.beta(**kwargs)
        gamma = self.gamma(**kwargs)
        if self.alpha == 1.0:
            bx_violation_prob = gamma + beta * self.horizon
        elif self.alpha * beta / (1 - self.alpha) <= 1:
            bx_violation_prob = 1 - (1 - gamma) * beta ** self.horizon
        else:
            bx_violation_prob = gamma * self.alpha ** (-self.horizon) + \
                (1 - self.alpha ** (-self.horizon)) * self.alpha * beta / (self.alpha - 1)

        if return_beta_gamma:
            return bx_violation_prob, beta, gamma

        return bx_violation_prob

    @torch.no_grad()
    def certify(self, **kwargs):
        """
        Certify that a trained barrier network is a valid barrier using the barrier loss
        Allow a small violation to account for potential numerical (FP) errors.
        :return: true if the barrier network is a valid barrier
        """
        return self.loss_barrier(**kwargs).item() <= 1.0e-10
