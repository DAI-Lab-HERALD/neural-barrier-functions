using SemialgebraicSets

# Convenience function to allow a possibly vectorize variable/polynomial/range, etc
macro maybe_vector(name, expr)
    return esc(:(const $name = Union{<:AbstractVector{$expr}, $expr}))
end