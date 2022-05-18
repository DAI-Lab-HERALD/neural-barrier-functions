using Revise
using DynamicPolynomials, MultivariatePolynomials

using BarrierFunctions

include("experiments/population.jl")
include("experiments/polynomial.jl")
include("visualization.jl")
include("utils.jl")

σ = 0.1 # Noise
H = 10  # Time horizon (max time)
system = polynomial_system
fig_folder = @name(polynomial_system)

succ, B, prob = verify(system, σ, H; B_deg = 8)

save_barrier(barriers[1], "models/polynomial_system_sos_barrier.json")
barrier_plot(fig_folder, i, barrier, 3.5)