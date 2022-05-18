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

X = @set (x[1] + 0.75)^2 ≤ 2.75^2 && (x[2] + 0.5)^2 ≤ 1.5^2
Xu = [@set((x[1] + 1.0)^2 + (x[2] + 1.0)^2 ≤ 0.16), @set((x[1] - 0.5)^2 ≤ 0.1^2 && (x[2] - 0.3)^2 ≤ 0.2^2), @set((x[1] - 0.6)^2 ≤ 0.2^2 && (x[2] - 0.2)^2 ≤ 0.1^2)]
Xi = [@set((x[1] - 1.5)^2 + x[2]^2 ≤ 0.25), @set((x[1] + 1.5)^2 ≤ 0.3^2 && x[2]^2 ≤ 0.1^2), @set((x[1] + 1.3)^2 ≤ 0.1^2 && (x[2] + 0.2)^2 ≤ 0.3^2)]

# Safe set would be the statement below, but since it contains disjunctions, it is not encodable in a semi-algebraic set.
# Xs = (x[1] + 1.0)^2 + (x[2] + 1.0)^2 ≥ 0.16 && ((x[1] - 0.5)^2 ≥ 0.1^2 || (x[2] - 0.3)^2 ≥ 0.2^2) && ((x[1] - 0.6)^2 ≥ 0.2^2 || (x[2] - 0.2)^2 ≥ 0.1^2)

# One solution is to make the expectation_constraint hold for the entire state space (i.e. also the safe set).
Xs = nothing  # When Xs is nothing, system automatically picks the state space for the expectation_constraint

# Another solution is to under-approximate rectangles by epllises (so that the entire safe set + a little extra is satisfies the expectation constraint)
# (x[1] - 0.5)^2 ≥ 0.1^2 || (x[2] - 0.3)^2 ≥ 0.2^2
ellipse1 = @set (x[1] - 0.5)^2 / 0.1^2 + (x[2] - 0.3)^2 / 0.2^2 ≥ 1.0

# (x[1] - 0.6)^2 ≥ 0.2^2 || (x[2] - 0.2)^2 ≥ 0.1^2
ellipse2 = @set (x[1] - 0.6)^2 / 0.2^2 + (x[2] - 0.2)^2 / 0.1^2 ≥ 1.0
Xs = @set((x[1] + 1.0)^2 + (x[2] + 1.0)^2 ≥ 0.16) ∩ ellipse1 ∩ ellipse2


partitioning = NoPartitioning()
polynomial_system = BarrierFunctions.autonomous_system(fx, x, z, X, Xu, Xi, Xs, partitioning)