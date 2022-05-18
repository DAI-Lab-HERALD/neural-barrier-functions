using Distributions
using SemialgebraicSets, DynamicPolynomials, MultivariatePolynomials

abstract type Expectation end
struct AnalyticExpectation <: Expectation end



function auxiliary_domain(yz_pairs, system::System{V, <:ExactDynamics, C, P}, p, 𝔼) where {V, C, P}
    z, fx = noise(system), next_state(system, p)
    domain = mapreduce(∩, yz_pairs) do (y_sample, z_sample)
        x_next = subs(fx, z => z_sample)
        if 𝔼.force_semialgebraic
            basicsemialgebraicset(FullSpace(), vcat(y_sample - x_next, x_next - y_sample))
        else
            algebraicset(x_next - y_sample)
        end
    end

    return domain
end

function auxiliary_domain(yz_pairs, system::System{V, <:BoundDynamics, C, P}, p, _) where {V, C, P}
    z, (lower, upper) = noise(system), next_state(system, p)
    domain = mapreduce(∩, yz_pairs) do (y_sample, z_sample)
        x_next_lower = subs(lower, z => z_sample)
        x_next_upper = subs(upper, z => z_sample)
        basicsemialgebraicset(FullSpace(), vcat(y_sample - x_next_lower, x_next_upper - y_sample))
    end

    return domain
end

function expectation(::AnalyticExpectation, system::System{V, <:ExactDynamics, C, P}, B, σ, p) where {V, C, P}
    x, z = state(system), noise(system)
    B_next = subs(B, x => next_state(system, p))

    𝔼Bfx = 0
    for term in terms(B_next)
        for z_i in z
            z_power = degree(term, z_i)

            if z_power == 0
                z_factor = 1
            elseif z_power % 2 == 1
                z_factor = 0
            else
                z_factor = prod(1:2:(z_power-1)) * σ^z_power
            end

            # Replace z^n with z_factor by first replacing z with 1 and then multiplying by z_factor
            term = subs(term, z_i => 1) * z_factor
        end
        𝔼Bfx += term
    end

    return 𝔼Bfx, nothing, FullSpace()
end