using Test
using SyntheticscRNAseq
using Random
using Statistics

@testset "SyntheticscRNAseq" begin
    include("test_ssa_analytical.jl")
    include("test_ssa_vs_catalyst.jl")
    include("test_tauleap_convergence.jl")
    include("test_binomial_vs_poisson.jl")
    include("test_cle_vs_tauleap.jl")
    include("test_capture.jl")
    include("test_population.jl")
    include("test_growth_feedback.jl")
    include("test_analytical_moments.jl")
    include("test_telegraph_distribution.jl")
    include("test_gpu_parity.jl")
end
