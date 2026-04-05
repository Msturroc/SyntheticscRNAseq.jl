module SyntheticscRNAseq

using Random
using Statistics
using LinearAlgebra
using Distributions
using StatsBase

# ── Core types ───────────────────────────────────────────────────
include("network.jl")
include("kinetics.jl")
include("population.jl")

# ── Algorithms ───────────────────────────────────────────────────
include("algorithms/ssa.jl")
include("algorithms/tauleap_fixed.jl")
include("algorithms/tauleap_binomial.jl")
include("algorithms/tauleap_midpoint.jl")
include("algorithms/cle.jl")
include("algorithms/cle_optimized.jl")
include("algorithms/tauleap_binomial_optimized.jl")

# ── Output models ────────────────────────────────────────────────
include("capture.jl")

# ── Validation utilities ─────────────────────────────────────────
include("validate.jl")

# ── Analytical solutions ─────────────────────────────────────────
include("analytical.jl")

# ── Default algorithm selection ──────────────────────────────────
include("default_algorithm.jl")

# ── Exports ──────────────────────────────────────────────────────

# Types
export GeneNetwork, CoopEdge, RedunEdge
export KineticParams
export PopulationConfig, CaptureModel

# Algorithms
export SSA, PoissonTauLeap, BinomialTauLeap, MidpointTauLeap, CLE
export CLEFast, BinomialTauLeapFast

# Main interface
export simulate, default_algorithm

# Network utilities
export hill_activation, hill_repression, hill_regulate
export regulatory_input, precompute_hill_matrices
export n_species
export steady_state, sample_network, is_bursty

# Capture
export apply_capture

# Validation
export ks_statistic, ks_critical_value, compare_moments, convergence_rate

# Analytical solutions
export telegraph_logpmf, telegraph_pmf, telegraph_distribution
export telegraph_mean, telegraph_variance, telegraph_fano

# Population dynamics
export PopulationState, initialize_population
export grow_volumes!, division_check!, extract_snapshot
export simulate_with_state

# GPU batch interface (implemented in CUDA extension)
function simulate_gpu_batch end
export simulate_gpu_batch

# Convenience
export simulate_birth_death

end # module
