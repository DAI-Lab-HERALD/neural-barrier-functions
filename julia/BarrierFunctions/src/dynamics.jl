
abstract type AbstractDynamics end
abstract type ExactDynamics <: AbstractDynamics end
abstract type BoundDynamics <: AbstractDynamics end

struct VectorDynamics{V} <: ExactDynamics
    fx::AbstractVector{V}
end

struct VectorBoundDynamics{V} <: BoundDynamics end


dynamics(fx::AbstractPolynomialLike) = dynamics([fx])
dynamics(fx::AbstractVector{V}) where {V} = VectorDynamics(fx)

next_state(d::VectorDynamics{V}, p) where {V} = d.fx
# next_state(d::VectorBoundDynamics{V}, p) where {V} = missing