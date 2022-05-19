module BarrierFunctions
    using MultivariatePolynomials
    const MP = MultivariatePolynomials

    const VectorAbstractVariable = AbstractVector{<:MP.AbstractVariable}
    const VectorAPL{T} = AbstractVector{<:AbstractPolynomial{T}}

    include("util.jl")

    include("partitioning.jl")
    export NoPartitioning, EqualWidthPartitioning, IndexedEqualWidthPartitioning, AbstractPartitioning

    include("variables.jl")
    export AbstractSystemVariables, variables, state, noise

    include("dynamics.jl")
    export AbstractDynamics, ExactDynamics, BoundDynamics, dynamics, next_state, VectorBoundDynamics

    include("system.jl")
    export System, autonomous_system
    export state_space, unsafe_set, initial_set, partitions

    include("expectation.jl")
    export Exception, AnalyticExpectation

    include("verify.jl")
    export verify
end