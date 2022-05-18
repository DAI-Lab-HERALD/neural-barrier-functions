module BarrierFunctions
    using MultivariatePolynomials
    const MP = MultivariatePolynomials

    const VectorAbstractVariable = AbstractVector{<:MP.AbstractVariable}
    const VectorAPL{T} = AbstractVector{<:AbstractPolynomial{T}}

    include("util.jl")
    include("substitute.jl")

    include("polytope.jl")
    export @polytope, inequalities, polytope

    include("partitioning.jl")
    export NoPartitioning, EqualWidthPartitioning

    include("variables.jl")
    export AbstractSystemVariables, variables, state, noise

    include("dynamics.jl")
    export AbstractDynamics, ExactDynamics, BoundDynamics, dynamics, next_state

    include("system.jl")
    export System, autonomous_system
    export state_space, unsafe_set, initial_set, partitions

    include("expectation.jl")
    export Exception, AnalyticExpectation

    include("sparsity.jl")
    include("verify.jl")
    export verify
end