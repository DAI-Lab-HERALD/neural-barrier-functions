abstract type AbstractSystemVariables end

struct VectorSystemVariables{V} <: AbstractSystemVariables
    x::AbstractVector{V}
    z::AbstractVector{V}
end

variables(x, z::MP.AbstractVariable) = variables(x, [z])
variables(x::MP.AbstractVariable, z::AbstractVector{V}) where {V} = variables([x], z)
variables(x::AbstractVector{V}, z::AbstractVector{V}) where {V} = VectorSystemVariables(x, z)


state(v::VectorSystemVariables{V}) where {V} = v.x
noise(v::VectorSystemVariables{V}) where {V} = v.z
