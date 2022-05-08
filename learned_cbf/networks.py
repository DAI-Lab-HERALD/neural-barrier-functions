import math

import torch
from bound_propagation import Residual, BoundModule, LinearBounds, IntervalBounds, HyperRectangle, Sub, BoundActivation, \
    Clamp, Parallel, Add
from bound_propagation.activation import assert_bound_order, BoundSigmoid, crown_backward_act_jit
from bound_propagation.linear import crown_backward_linear_jit
from torch import nn, Tensor


class Mean(nn.Module):
    def __init__(self, subnetwork):
        super().__init__()

        self.subnetwork = subnetwork

    def forward(self, x):
        return self.subnetwork(x).mean(dim=0)


class BoundMean(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.subnetwork = factory.build(module.subnetwork)

    @property
    def need_relaxation(self):
        return self.subnetwork.need_relaxation

    def clear_relaxation(self):
        self.subnetwork.clear_relaxation()

    def backward_relaxation(self, region):
        return self.subnetwork.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        subnetwork_bounds = self.subnetwork.crown_backward(linear_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (subnetwork_bounds.lower[0].mean(dim=0), subnetwork_bounds.lower[1].mean(dim=0))

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (subnetwork_bounds.upper[0].mean(dim=0), subnetwork_bounds.upper[1].mean(dim=0))

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        subnetwork_bounds = self.subnetwork.ibp_forward(bounds, save_relaxation=save_relaxation)
        return IntervalBounds(bounds.region, subnetwork_bounds.lower.mean(dim=0), subnetwork_bounds.upper.mean(dim=0))

    def propagate_size(self, in_size):
        return self.subnetwork.propagate_size(in_size)


class BetaNetwork(nn.Module):
    def __init__(self, dynamics, barrier):
        super().__init__()

        self.dynamics_barrier = Mean(nn.Sequential(dynamics, barrier))
        self.barrier = barrier

    def forward(self, x):
        return self.dynamics_barrier(x) - self.barrier(x)


class BoundBetaNetwork(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_dynamics_barrier = factory.build(module.dynamics_barrier)
        self.bound_barrier = factory.build(module.barrier)

    @property
    def need_relaxation(self):
        return self.bound_dynamics_barrier.need_relaxation or self.bound_barrier.need_relaxation

    def clear_relaxation(self):
        self.bound_dynamics_barrier.clear_relaxation()
        self.bound_barrier.clear_relaxation()

    def backward_relaxation(self, region):
        if self.bound_dynamics_barrier.need_relaxation:
            return self.bound_dynamics_barrier.backward_relaxation(region)
        else:
            assert self.bound_barrier.need_relaxation
            return self.bound_barrier.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        input_bounds = LinearBounds(
            linear_bounds.region,
            (linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None,
        )
        linear_bounds1 = self.bound_dynamics_barrier.crown_backward(input_bounds)

        # We can only do this inversion for a "Sub" module because we know nothing is ahead (after in the network).
        input_bounds = LinearBounds(
            linear_bounds.region,
            (linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None,
            (linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
        )
        linear_bounds2 = self.bound_barrier.crown_backward(input_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (linear_bounds1.lower[0] - linear_bounds2.upper[0], linear_bounds1.lower[1] - linear_bounds2.upper[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (linear_bounds1.upper[0] - linear_bounds2.lower[0], linear_bounds1.upper[1] - linear_bounds2.lower[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        bounds1 = self.bound_dynamics_barrier.ibp_forward(bounds, save_relaxation=save_relaxation)
        bounds2 = self.bound_barrier.ibp_forward(bounds, save_relaxation=save_relaxation)

        return IntervalBounds(
            bounds.region,
            bounds1.lower - bounds2.upper,
            bounds1.upper - bounds2.lower
        )

    def propagate_size(self, in_size):
        out_size1 = self.bound_dynamics_barrier.propagate_size(in_size)
        out_size2 = self.bound_barrier.propagate_size(in_size)

        assert out_size1 == out_size2

        return out_size1


class AqiNetwork(nn.Module):
    def __init__(self, A_qi, b_qi=None):
        super().__init__()

        self.register_buffer('A_qi_bot', A_qi[0])
        self.register_buffer('A_qi_top', A_qi[1])

        if b_qi is None:
            self.register_buffer('b_qi_bot', None)
            self.register_buffer('b_qi_top', None)
        else:
            self.register_buffer('b_qi_bot', b_qi[0].unsqueeze(1))
            self.register_buffer('b_qi_top', b_qi[1].unsqueeze(1))

    def forward(self, x: Tensor) -> Tensor:
        x = x.matmul(self.A_qi_bot.transpose(-1, -2))
        if self.b_qi_bot is not None:
            x = x + self.b_qi_bot
        return x


class BoundAqiNetwork(BoundModule):
    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds):
        if linear_bounds.lower is None:
            lower = None
        else:
            lower = crown_backward_linear_jit(self.module.A_qi_bot, self.module.b_qi_bot, linear_bounds.lower[0])
            lower = (lower[0], lower[1] + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = crown_backward_linear_jit(self.module.A_qi_top, self.module.b_qi_top, linear_bounds.upper[0])
            upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        center, diff = bounds.center, bounds.width / 2

        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        A_qi_bot = self.module.A_qi_bot.transpose(-1, -2)
        lower = (center.matmul(A_qi_bot) - diff.matmul(A_qi_bot.abs())).squeeze(-2)
        if self.b_qi_bot is not None:
            lower = lower + self.module.b_qi_bot

        A_qi_top = self.module.A_qi_top.transpose(-1, -2)
        upper = (center.matmul(A_qi_top) + diff.matmul(A_qi_top.abs())).squeeze(-2)
        if self.b_qi_top is not None:
            upper = upper + self.module.b_qi_top

        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        return self.module.A_qi_top.size(-2)


class Constant(nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return self.constant * x


class BoundConstant(BoundModule):

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds):
        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (self.module.constant * linear_bounds.lower[0], linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (self.module.constant * linear_bounds.upper[0], linear_bounds.upper[1])

        neg = self.module.constant < 0.0
        if torch.any(neg):
            assert lower is not None and upper is not None, 'bound_lower=False and bound_upper=False cannot be used with a negative constant'
            lower = torch.where(neg, upper[0], lower[0]), lower[1]
            upper = torch.where(neg, lower[0], upper[0]), upper[1]

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        return IntervalBounds(bounds.region, self.module.constant * bounds.lower, self.module.constant * bounds.upper)

    def propagate_size(self, in_size):
        return in_size


class Exp(nn.Module):
    def forward(self, x):
        return x.exp()


class BoundExp(BoundActivation):
    def func(self, x):
        return x.exp()

    def derivative(self, x):
        return x.exp()

    @assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper
        zero_width = torch.isclose(lower, upper, rtol=0.0, atol=1e-8)
        other = ~zero_width

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        lower_act, upper_act = self.func(lower), self.func(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self.func(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        add_linear(self.alpha_upper, self.beta_upper, mask=other, a=slope, x=lower, y=lower_act)
        add_linear(self.alpha_lower, self.beta_lower, mask=other, a=d_prime, x=d, y=d_act)


class QBound(nn.Module):
    def __init__(self, dynamics_network, q_bounds, scale, factor=2.0):
        super(QBound, self).__init__()

        self.dynamics_network = dynamics_network
        self.q_bounds = q_bounds.unsqueeze(1)
        self.scale = factor * scale

    def forward(self, x):
        x = (x - self.q_bounds) / self.scale
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class BoundQBound(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_dynamics = factory.build(module.dynamics_network)

    @property
    def need_relaxation(self):
        return self.bound_dynamics.need_relaxation

    def clear_relaxation(self):
        self.bound_dynamics.clear_relaxation()

    def backward_relaxation(self, region):
        return self.bound_dynamics.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        assert linear_bounds.lower is not None and linear_bounds.upper is not None, 'bound_lower=False and bound_upper=False cannot be used with QBound'

        input_bounds = LinearBounds(linear_bounds.region,
            ((linear_bounds.upper[0] / self.module.scale).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(linear_bounds.lower[1])),
            ((linear_bounds.lower[0] / self.module.scale).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(linear_bounds.upper[1])),
        )

        dynamics_bounds = self.bound_dynamics.crown_backward(input_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            lowerA = -dynamics_bounds.upper[0]
            lower_bias = (self.module.q_bounds / self.module.scale).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            lower_bias = lower_bias.unsqueeze(-2).matmul(linear_bounds.lower[0].transpose(-1, -2)).squeeze(-2)
            lower = (lowerA, -dynamics_bounds.upper[1] + linear_bounds.lower[1] + lower_bias)

        if linear_bounds.upper is None:
            upper = None
        else:
            upperA = -dynamics_bounds.lower[0]
            upper_bias = (self.module.q_bounds / self.module.scale).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            upper_bias = upper_bias.unsqueeze(-2).matmul(linear_bounds.upper[0].transpose(-1, -2)).squeeze(-2)
            upper = (upperA, -dynamics_bounds.lower[1] + linear_bounds.upper[1] + upper_bias)

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        lower = ((self.module.q_bounds - bounds.upper) / self.module.scale).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        upper = ((self.module.q_bounds - bounds.lower) / self.module.scale).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        return in_size


class Square(nn.Module):
    def forward(self, x):
        return x ** 2


class BoundSquare(BoundActivation):
    def func(self, x):
        return x ** 2

    def derivative(self, x):
        return 2 * x

    @assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper
        zero_width = torch.isclose(lower, upper, rtol=0.0, atol=1e-8)
        other = ~zero_width

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        lower_act, upper_act = self(lower[zero_width]), self(upper[zero_width])
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, torch.min(lower_act, upper_act)
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, torch.max(lower_act, upper_act)

        lower_act, upper_act = self.func(lower), self.func(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self.func(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        add_linear(self.alpha_upper, self.beta_upper, mask=other, a=slope, x=lower, y=lower_act)
        add_linear(self.alpha_lower, self.beta_lower, mask=other, a=d_prime, x=d, y=d_act)


class Sum(nn.Module):
    def __init__(self, subnetwork):
        super().__init__()

        self.subnetwork = subnetwork

    def forward(self, x):
        return self.subnetwork(x).mean(dim=0)


class BoundSum(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.subnetwork = factory.build(module.subnetwork)

    @property
    def need_relaxation(self):
        return self.subnetwork.need_relaxation

    def clear_relaxation(self):
        self.subnetwork.clear_relaxation()

    def backward_relaxation(self, region):
        return self.subnetwork.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        subnetwork_bounds = self.subnetwork.crown_backward(linear_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (subnetwork_bounds.lower[0].sum(dim=0), subnetwork_bounds.lower[1].sum(dim=0))

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (subnetwork_bounds.upper[0].sum(dim=0), subnetwork_bounds.upper[1].sum(dim=0))

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        subnetwork_bounds = self.subnetwork.ibp_forward(bounds, save_relaxation=save_relaxation)
        return IntervalBounds(bounds.region, subnetwork_bounds.lower.sum(dim=0), subnetwork_bounds.upper.sum(dim=0))

    def propagate_size(self, in_size):
        return self.subnetwork.propagate_size(in_size)


class TruncatedGaussianExpectation(Sum):
    def __init__(self, dynamics_network, A_qi, scale, q_bounds):
        super().__init__(
            nn.Sequential(
                Sub(
                    nn.Sequential(
                        QBound(dynamics_network, q_bounds[0], scale),
                        Square(),
                        Constant(torch.full_like(scale, -2)),
                        Exp()
                    ),
                    nn.Sequential(
                        QBound(dynamics_network, q_bounds[1], scale),
                        Square(),
                        Constant(torch.full_like(scale, -2)),
                        Exp()
                    ),
                ),
                Constant(scale / math.sqrt(2 * math.pi)),
                AqiNetwork(A_qi)
            )
        )


class Erf(nn.Module):
    def forward(self, x):
        return torch.erf(x)


class BoundErf(BoundSigmoid):
    def func(self, x):
        return torch.erf(x)

    def derivative(self, x):
        x_squared = x ** 2
        return (2.0 / math.sqrt(math.pi)) * (-x_squared).exp()


class IgnoreZeroScale(nn.Module):
    def __init__(self, dynamics_network, scale, q_bounds):
        super().__init__()

        self.dynamics_network = dynamics_network
        self.scale = scale
        self.q_bounds = q_bounds

    def forward(self, x):
        v_bounds = self.q_bounds[0].unsqueeze(1) - x, self.q_bounds[1].unsqueeze(1) - x
        return torch.where(self.scale > 0.0, x, ((v_bounds[0] <= 0.0) & (0.0 <= v_bounds[1])).to(x.dtype))


class BoundIgnoreZeroScale(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_dynamics = factory.build(module.dynamics_network)

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds):
        f_bounds = self.bound_dynamics.crown(linear_bounds.region).concretize()
        v_bounds_under, v_bounds_over = self.v_bounds(f_bounds)
        zero_scale = (self.module.scale == 0.0)

        if linear_bounds.lower is None:
            lower = None
        else:
            lowerA = linear_bounds.lower[0].clone()
            lowerA[..., zero_scale] = 0.0
            lower_bias = linear_bounds.lower[0].matmul(self.ignore_zero_scale_lower(v_bounds_under).unsqueeze(-1))
            lower = (lowerA, linear_bounds.lower[1] + lower_bias.squeeze(-1))

        if linear_bounds.upper is None:
            upper = None
        else:
            upperA = linear_bounds.upper[0].clone()
            upperA[..., zero_scale] = 0.0
            upper_bias = linear_bounds.upper[0].matmul(self.ignore_zero_scale_upper(v_bounds_over).unsqueeze(-1))
            upper = (upperA, linear_bounds.upper[1] + upper_bias.squeeze(-1))

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        f_bounds = self.bound_dynamics.ibp(bounds.region)
        lower, upper = self.ignore_zero_scale(f_bounds, bounds.lower, bounds.upper)

        return IntervalBounds(bounds.region, lower, upper)

    def v_bounds(self, dynamics_interval):
        dynamics_lower, dynamics_upper = dynamics_interval.lower, dynamics_interval.upper
        q_lower, q_upper = self.module.q_bounds[0].unsqueeze(1), self.module.q_bounds[1].unsqueeze(1)

        v_bounds_under = q_lower - dynamics_lower, q_upper - dynamics_upper
        v_bounds_over = q_lower - dynamics_upper, q_upper - dynamics_lower

        return v_bounds_under, v_bounds_over

    def ignore_zero_scale(self, dynamics_interval, lower, upper):
        v_bounds_under, v_bounds_over = self.v_bounds(dynamics_interval)
        zero_scale = (self.module.scale == 0.0)

        if lower.dim() != v_bounds_under[0].dim():
            lower = lower.unsqueeze(0).expand(*v_bounds_under[0].size())
            upper = upper.unsqueeze(0).expand(*v_bounds_over[0].size())

        lower[..., zero_scale] = self.ignore_zero_scale_lower(v_bounds_under)[..., zero_scale]
        upper[..., zero_scale] = self.ignore_zero_scale_upper(v_bounds_over)[..., zero_scale]

        return lower, upper

    def ignore_zero_scale_lower(self, v_bounds_under):
        zero_scale = (self.module.scale == 0.0)
        return (zero_scale & (v_bounds_under[0] <= 0.0) & (0.0 <= v_bounds_under[1]) & (v_bounds_under[0] >= v_bounds_under[1])).float()

    def ignore_zero_scale_upper(self, v_bounds_over):
        zero_scale = (self.module.scale == 0.0)
        return (zero_scale & (v_bounds_over[0] <= 0.0) & (0.0 <= v_bounds_over[1])).float()

    def propagate_size(self, in_size):
        out_size = self.bound_dynamics.propagate_size(in_size)
        assert in_size == out_size
        return out_size


class ProbabilityNetwork(nn.Sequential):
    def __init__(self, dynamics_network, scale, q_bounds):
        super().__init__(
            Sub(
                nn.Sequential(
                    QBound(dynamics_network, q_bounds[1], scale, factor=math.sqrt(2.0)),
                    Erf()
                ),
                nn.Sequential(
                    QBound(dynamics_network, q_bounds[0], scale, factor=math.sqrt(2.0)),
                    Erf()
                )
            ),
            IgnoreZeroScale(dynamics_network, scale, q_bounds),
            Constant(torch.full_like(scale, 0.5)),
            Clamp(min=0.0, max=1.0)
        )


class Prod(nn.Module):
    def __init__(self, subnetwork):
        super().__init__()
        self.subnetwork = subnetwork

    def forward(self, x):
        return self.subnetwork(x).prod(dim=-1, keepdim=True)


class BoundProd(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_subnetwork = factory.build(module.subnetwork)
        self.out_size = None

    @property
    def need_relaxation(self):
        return self.bound_subnetwork.need_relaxation

    def backward_relaxation(self, region):
        return self.bound_subnetwork.backward_relaxation(region)

    def clear_relaxation(self):
        self.bound_subnetwork.clear_relaxation()

    def crown_backward(self, linear_bounds):
        input_bounds = self.initial_linear_bounds(linear_bounds.region, self.out_size)
        subnetwork_bounds = self.bound_subnetwork.crown_backward(input_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            acc = subnetwork_bounds.lower[0][..., 0, :], subnetwork_bounds.lower[1][..., 0]

            for i in range(1, subnetwork_bounds.lower[1].size(-1)):
                acc = self.combine_bounds(acc, (subnetwork_bounds.lower[0][..., i, :], subnetwork_bounds.lower[1][..., i]), linear_bounds.region)

            lower = (acc[0].unsqueeze(-2), acc[1].unsqueeze(-1) + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            acc = subnetwork_bounds.upper[0][..., 0, :], subnetwork_bounds.upper[1][..., 0]

            for i in range(1, subnetwork_bounds.upper[1].size(-1)):
                acc = self.combine_bounds(acc, (subnetwork_bounds.upper[0][..., i, :], subnetwork_bounds.upper[1][..., i]), linear_bounds.region, lower=False)

            upper = (acc[0].unsqueeze(-2), acc[1].unsqueeze(-1) + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def combine_bounds(self, acc, bounds, region, lower=True):
        linear = acc[0] * bounds[1].unsqueeze(-1) + bounds[0] * acc[1].unsqueeze(-1)
        bias = acc[1] * bounds[1]

        A_prime = acc[0].unsqueeze(-1).matmul(bounds[0].unsqueeze(-2))
        A = (A_prime + A_prime.transpose(-1, -2)) / 2.0

        # Gamma, U = torch.linalg.eigh(A)
        U, Gamma, _ = torch.linalg.svd(A)
        U_transpose = U.transpose(-1, -2)

        center, diff = region.center, region.width / 2.0
        W_center, W_diff = U_transpose.matmul(center.unsqueeze(-1)), U_transpose.abs().matmul(diff.unsqueeze(-1))
        y_bounds = IntervalBounds(region, (W_center - W_diff).squeeze(-1), (W_center + W_diff).squeeze(-1))

        square_lower, square_upper = self.alpha_beta(y_bounds)
        if lower:
            square_lower, square_upper = square_upper, square_lower

        A_x = (Gamma * torch.where(Gamma < 0.0, square_lower[0], square_upper[0])).unsqueeze(-2).matmul(U_transpose).squeeze(-2)
        b_x = (Gamma * torch.where(Gamma < 0.0, square_lower[1], square_upper[1])).sum(dim=-1)

        return A_x + linear, b_x + bias

    def square(self, x):
        return x ** 2

    def square_derivative(self, x):
        return 2 * x

    @assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper
        zero_width = torch.isclose(lower, upper, rtol=0.0, atol=1e-8)
        other = ~zero_width

        alpha_lower, beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        alpha_upper, beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        lower_act, upper_act = self.square(lower[zero_width]), self.square(upper[zero_width])
        alpha_lower[zero_width], beta_lower[zero_width] = 0, torch.min(lower_act, upper_act)
        alpha_upper[zero_width], beta_upper[zero_width] = 0, torch.max(lower_act, upper_act)

        lower_act, upper_act = self.square(lower), self.square(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self.square(d)
        d_prime = self.square_derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        add_linear(alpha_upper, beta_upper, mask=other, a=slope, x=lower, y=lower_act)
        add_linear(alpha_lower, beta_lower, mask=other, a=d_prime, x=d, y=d_act)

        return (alpha_lower, beta_lower), (alpha_upper, beta_upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        bounds = self.bound_subnetwork.ibp_forward(bounds, save_relaxation=save_relaxation)
        lower, upper = bounds.lower[..., 0], bounds.upper[..., 0]

        for i in range(1, bounds.lower.size(-1)):
            dim_lower, dim_upper = bounds.lower[..., i], bounds.upper[..., i]
            candidates = lower * dim_lower, lower * dim_upper, upper * dim_lower, upper * dim_upper
            lower = torch.min(torch.min(*candidates[:2]), torch.min(*candidates[2:]))
            upper = torch.max(torch.max(*candidates[:2]), torch.max(*candidates[2:]))

        return IntervalBounds(bounds.region, lower.unsqueeze(-1), upper.unsqueeze(-1))

    def propagate_size(self, in_size):
        self.out_size = self.bound_subnetwork.propagate_size(in_size)
        return 1


class AqiDynamicsNetwork(nn.Sequential):
    def __init__(self, dynamics_network, A_qi, b_qi):
        super(AqiDynamicsNetwork, self).__init__(
            dynamics_network,
            AqiNetwork(A_qi, b_qi)
        )


class NominalSystemBarrier(Sum):
    def __init__(self, dynamics_network, A_qi, b_qi, scale, q_bounds):
        super().__init__(
            Prod(
                Parallel(
                    AqiDynamicsNetwork(dynamics_network, A_qi, b_qi),
                    ProbabilityNetwork(dynamics_network, scale, q_bounds)
                )
            )
        )


class ExpectationBounds(Sub):
    def __init__(self, barrier, dynamics_network, A_qi, b_qi, scale, q_bounds):
        super().__init__(
            Add(
                NominalSystemBarrier(dynamics_network, A_qi, b_qi, scale, q_bounds),
                TruncatedGaussianExpectation(dynamics_network, A_qi, scale, q_bounds)
            ),
            barrier
        )


class FCNNBarrierNetwork(nn.Sequential):
    activation_class_mapping = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh
    }

    def __init__(self, network_config):
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


class ResidualBarrierNetwork(nn.Sequential):
    activation_class_mapping = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh
    }

    def __init__(self, network_config):
        assert network_config['type'] == 'residual'

        activation_class = self.activation_class_mapping[network_config['activation']]

        # First hidden layer (since it requires another input dimension
        layers = [nn.Linear(network_config['input_dim'], network_config['hidden_nodes']), activation_class()]

        # Other hidden layers
        for _ in range(network_config['hidden_layers'] - 1):
            layers.append(
                Residual(nn.Sequential(
                    nn.Linear(network_config['hidden_nodes'], network_config['hidden_nodes']),
                    activation_class()
                ))
            )

        # Output layer (no activation)
        layers.append(nn.Linear(network_config['hidden_nodes'], 1))

        super().__init__(*layers)
