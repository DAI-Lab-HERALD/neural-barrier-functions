using MultivariatePolynomials, DynamicPolynomials, SemialgebraicSets
using HomotopyContinuation

using BarrierFunctions

####################
# Polynomial system
####################
dt = 0.1

# Before designing system further, we want to prove the origin is an equilibrium
@var x₁ x₂ x₃ x₄
fx = HomotopyContinuation.System([x₁, x₂, x₃, x₄] + dt * [
    x₁ + x₂ + x₃^3,
    x₁^2 + x₂ - x₃ - x₄,
    -x₁ + x₂^2 + x₃,
    -x₁ - x₂
])
result = solve(fx)
println(real_solutions(result))


@polyvar x[1:4] z
fx = x + dt * [
    x[1] + x[2] + x[3]^3,
    x[1]^2 + x[2] - x[3] - x[4],
    -x[1] + x[2]^2 + x[3],
    -x[1] - x[2]
]

fxz = fx + [dt * z, 0, 0, 0]

X = @set x[1]^2 ≤ 1.5^2 && x[2]^2 ≤ 1.5^2 && x[3]^2 ≤ 1.5^2 && x[4]^2 ≤ 1.5^2

Xi = @set x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 ≤ 0.2^2
Xu = X ∩ @set x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 ≥ 1.0^2
Xs = @set x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 ≤ 1.0^2


partitioning = NoPartitioning()
polynomial_4d_system = BarrierFunctions.autonomous_system(fx, x, z, X, Xu, Xi, Xs, partitioning)