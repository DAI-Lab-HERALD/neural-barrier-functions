using JuMP, MosekTools, SumOfSquares, Hypatia
using SemialgebraicSets, DynamicPolynomials, MultivariatePolynomials

function verify(system::System, σ::Number, H::Int; B_deg::Int = 4, 𝔼::Expectation = AnalyticExpectation())
    @assert(σ >= 0, "Standard deviation must be greater or equal to zero (is $σ)")
    @assert(H >= 1, "Horizon N must be greater than or equal to one (is $H)")
    @assert(B_deg >= 1, "Barrier polynomial degree must be greater than or equal to 1 (is $B_deg)")

    solver = optimizer_with_attributes(Mosek.Optimizer, 
        "QUIET" => false, 
        "MSK_DPAR_INTPNT_CO_TOL_MU_RED" => 1e-6,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => 1e-4)
    model = SOSModel(solver)

    # Decision variables
    @variable(model, γ, lower_bound = 0, upper_bound = 1)
    @variable(model, β, lower_bound = 0, upper_bound = 1)
    # This constraint is technically not necessary but it helps produce
    # valid probabilities and it reduces solver time.
    @constraint(model, γ + β * H <= 1)

    # Constraints
    monos = monomials(state(system), 0:B_deg)
    @variable(model, B, Poly(MonomialBasis(monos)))

    @constraint(model, B >= 0, domain = state_space(system))
    unsafe_constraint!(model, system, B)
    initial_constraint!(model, system, B, γ)

    expectation_constraint!(model, system, B, β, σ, 𝔼)

    # Objective
    @objective(model, Min, γ + β * H)

    # Solve
    JuMP.optimize!(model)
    # println(JuMP.solution_summary(model))

    succ = JuMP.primal_status(model) == FEASIBLE_POINT

    B = value(B)
    β = value(β)
    γ = value(γ)

    println("γ $γ, beta $β")

    # Alternatively use B(x_0) for γ if x_0 is absolutely known
    prob = γ + β * H

    println("termination status $(JuMP.termination_status(model)), primal status $(JuMP.primal_status(model)), prob $prob")

    return succ, B, prob
end

function initial_constraint!(model, system, B, γ)
    for initial_subset in initial_set(system)
        println("Initial subset ", initial_subset)
        @constraint(model, B <= γ, domain = initial_subset)
    end
end

function unsafe_constraint!(model, system, B)
    for unsafe_subset in unsafe_set(system)
        println("Unsafe subset ", unsafe_subset)
        @constraint(model, B >= 1, domain = unsafe_subset)
    end
end

function expectation_constraint!(model, system, B, β, σ, 𝔼)
    for p in partitions(system)
        # TODO Do not add any constraint if intersection between safe set and partition is empty
        # We may need to solve a Positivstellensatz problem
        println("Partition $p")

        partition_constraint!(model, system, B, β, σ, 𝔼, p)
    end
end

function partition_constraint!(model, system, B, β, σ, 𝔼, p)
    𝔼Bfx, aux_domain = expectation(𝔼, system, B, σ, p)
    full_domain = domain(p) ∩ aux_domain

    @constraint(model, 𝔼Bfx <= B + β, domain = full_domain)
end
