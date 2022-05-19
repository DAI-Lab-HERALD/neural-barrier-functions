using MultivariatePolynomials, DynamicPolynomials, SemialgebraicSets
using LinearAlgebra
using JSON

using BarrierFunctions

@polyvar x[1:3] z

X = @set x[1]^2 ≤ 2^2 && x[2]^2 ≤ 2^2 && x[3]^2 ≤ (π \ 2)^2
Xu = [@set(x[1]^2 ≥ 1.9), @set(x[2]^2 ≥ 1.9)]
Xi = @set x[1] == -0.95 && x[2] == 0.0 && x[3] == 0.0
Xs = @set x[1]^2 ≤ 1.9 && x[2]^2 ≤ 1.9

bounds = JSON.parsefile("models/dubin_bounds.json")

struct IndexPartitioning <: AbstractPartitioning
    num_partitions::Integer
end

function BarrierFunctions.partition_iterator(partitioning::IndexPartitioning)    
    return 1:partitioning.num_partitions
end

function BarrierFunctions.domain(p::Integer) 
    region_lower = bounds["region_lower"][p]
    region_upper = bounds["region_upper"][p]
    
    return basicsemialgebraicset(FullSpace(), vcat(x - region_lower, region_upper - x))
end

function BarrierFunctions.next_state(d::VectorBoundDynamics{V}, p) where {V} 
    lowerA = map((x) -> convert(Vector{Float32}, x), bounds["lowerA"][p])
    lower_bias = convert(Vector{Float32}, bounds["lower_bias"][p])
    upperA = map((x) -> convert(Vector{Float32}, x), bounds["upperA"][p])
    upper_bias = convert(Vector{Float32}, bounds["upper_bias"][p])

    zero_variables = 0 * x[1] + 0 * x[2] + 0 * x[3]

    x1_lower = zero_variables + dot(lowerA[1], x) + lower_bias[1]
    x2_lower = zero_variables + dot(lowerA[2], x) + lower_bias[2]
    x3_lower = zero_variables + dot(lowerA[3], x) + lower_bias[3] - z

    x1_upper = zero_variables + dot(upperA[1], x) + upper_bias[1]
    x2_upper = zero_variables + dot(upperA[2], x) + upper_bias[2]
    x3_upper = zero_variables + dot(upperA[3], x) + upper_bias[3] - z

    return vcat(x1_lower, x2_lower, x3_lower), vcat(x1_upper, x2_upper, x3_upper)
end

partitioning = IndexPartitioning(bounds["num_partitions"])
vars = BarrierFunctions.variables(x, z)
dubin_system = System(X, Xu, Xi, Xs, vars, VectorBoundDynamics{typeof(vars)}(), partitioning)
