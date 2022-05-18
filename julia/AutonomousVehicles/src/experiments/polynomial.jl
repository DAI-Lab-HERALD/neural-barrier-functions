using MultivariatePolynomials, DynamicPolynomials, SemialgebraicSets

using BarrierFunctions

@polyvar x[1:2] z

####################
# Polynomial system
####################
dt = 0.1

# f(x) = [
#     x[2] + z,
#     (x[1]^3) / 3.0 - x[1] - x[2]
# ]

# k1 = dt * f(x)
# k2 = dt * f(x + k1 / 2)
# k3 = dt * f(x + k2 / 2)
# k4 = dt * f(x + k3)

# function _remove_practical_zeros!(p::Polynomial)
#     zeroidx = Int[]
#     for (i, α) in enumerate(p.a)
#         if abs(α) <= 1e-10
#             push!(zeroidx, i)
#         end
#     end
#     if !isempty(zeroidx)
#         deleteat!(p.a, zeroidx)
#         deleteat!(p.x.Z, zeroidx)
#     end
# end

# fx = x + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
# # _remove_practical_zeros!(fx[1])
# # _remove_practical_zeros!(fx[2])
# println(fx)

fx = [
    x[1] + dt * (x[2] + z),
    x[2] + dt * ((x[1]^3) / 3.0 - x[1] - x[2])
]
println(fx)

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
polynomial_system = autonomous_system(fx, x, z, X, Xu, Xi, Xs, partitioning)

