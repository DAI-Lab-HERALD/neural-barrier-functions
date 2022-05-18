using MultivariatePolynomials, DynamicPolynomials, SemialgebraicSets

using BarrierFunctions

@polyvar x[1:2] z

# x[1] = juveniles
# x[2] = adults

####################
# Circle
####################
fertility_rate = 0.4
survival_juvenile = 0.3
survival_adult = 0.8

fx = [
    fertility_rate * x[2],
    survival_juvenile * x[1] + survival_adult * x[2] + z
]
println(fx)

X = @set x[1]^2 ≤ 3^2 && x[2]^2 ≤ 3^2
Xu = @set x[1]^2 + x[2]^2 ≥ 2^2 && x[1]^2 ≤ 3^2 && x[2]^2 ≤ 3^2
Xi = @set x[1]^2 + x[2]^2 ≤ 1.5^2
Xs = @set x[1]^2 + x[2]^2 ≤ 2^2

partitioning = NoPartitioning()
circle_population_system = autonomous_system(fx, x, z, X, Xu, Xi, Xs, partitioning)
