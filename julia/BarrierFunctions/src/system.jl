using SemialgebraicSets

struct System{V <: AbstractSystemVariables, D <: AbstractDynamics, P <: AbstractPartitioning}
    X::AbstractSemialgebraicSet
    Xu::AbstractVector{<:AbstractSemialgebraicSet}
    Xi::AbstractVector{<:AbstractSemialgebraicSet}
    Xs::Union{AbstractSemialgebraicSet, Nothing}

    v::V
    d::D
    p::P
end

autonomous_system(fx, x, z, X, Xu, Xi, Xs, p = NoPartitioning()) = System(X, Xu, Xi, Xs, variables(x, z), dynamics(fx), p)

System(X, Xu::AbstractSemialgebraicSet, Xi, Xs, v, d, p) = System(X, [Xu], Xi, Xs, v, d, p)
System(X, Xu, Xi::AbstractSemialgebraicSet, Xs, v, d, p) = System(X, Xu, [Xi], Xs, v, d, p)
System(X, Xu::AbstractSemialgebraicSet, Xi::AbstractSemialgebraicSet, Xs, v, d, p) = System(X, [Xu], [Xi], Xs, v, d, p)

state_space(s::System) = s.X
unsafe_set(s::System) = s.Xu
initial_set(s::System) = s.Xi
safe_set(s::System) = s.Xs

partitions(system::System{V, D, NoPartitioning}) where {V, D} = [isnothing(safe_set(system)) ? state_space(system) : safe_set(system)]
partitions(system::System{V, D, <:AbstractPartitioning}) where {V, D} = partition_iterator(system.p)

state(s::System) = state(s.v)
noise(s::System) = noise(s.v)
next_state(s::System, p) = next_state(s.d, p)
