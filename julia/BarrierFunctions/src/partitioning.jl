using SemialgebraicSets
using IterTools

abstract type AbstractPartitioning end

struct NoPartitioning <: AbstractPartitioning end

struct EqualWidthPartitioning{T, L <: Integer} <: AbstractPartitioning
    variables::AbstractVector{<:MP.AbstractVariable}
    iterators::AbstractVector{LinRange{T, L}}

    function EqualWidthPartitioning(variables::AbstractVector{<:MP.AbstractVariable}, iterators::AbstractVector{LinRange{T, L}}) where {T, L<:Integer}
        N = length(variables)
        M = length(iterators)
        @assert(N == M, "Number of variables $N must match number of LinRanges $M")

        return new{T, L}(variables, iterators)
    end
end

EqualWidthPartitioning(variables::MP.AbstractVariable, iterators...) = EqualWidthPartitioning([variables], iterators...)
EqualWidthPartitioning(variables::Tuple{Vararg{<:MP.AbstractVariable}}, iterators...) = EqualWidthPartitioning(vec(variables), iterators...)

function EqualWidthPartitioning(variables::AbstractVector{<:MP.AbstractVariable}, iterators::Vararg{LinRange})
    return EqualWidthPartitioning(variables, collect(iterators))
end

function partition_iterator(partitioning::EqualWidthPartitioning)
    slice_iterators = product(map(_slice, partitioning.iterators)...)
    partition_iterator = map(_convert_to_vec, slice_iterators)

    semialgebraic_sets = map((p) -> _semialgebraic_set(partitioning.variables, p), partition_iterator)
    
    return semialgebraic_sets
end

_slice(iterator) = partition(iterator, 2, 1)  # Tuple size = 2, Step size = 1
_convert_to_vec(slice) = vcat(first.(slice)...), vcat(last.(slice)...)

function _semialgebraic_set(x, (start, stop)::Tuple{T, T}) where {T}
    half_width = abs.(stop - start) / 2
    center = (start + stop) / 2
    x_center = x - center

    hyperrectangle = basicsemialgebraicset(FullSpace(), half_width.^2 - x_center.^2) # Implicit >= 0 for last parameter
    # hyperrectangle = basicsemialgebraicset(FullSpace(), vcat(half_width - x_center, x_center + half_width)) # Implicit >= 0 for last parameter

    return hyperrectangle
end
