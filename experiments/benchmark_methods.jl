#= ================================================================
   Benchmark: G=5 variable-kinetics regulated network

   Compares all methods across cell counts and simulation times.
   Tests equivalence (mean/variance agreement) and measures walltime.
   ================================================================ =#

using SyntheticscRNAseq
using Statistics
using Random
using Printf
using CUDA

# ── Build a G=5 network with mixed regulation ────────────────────
# 5 genes: gene 1 activated by gene 3, repressed by gene 5
#           gene 2 constitutive (high basal)
#           gene 3 activated by gene 4
#           gene 4 repressed by gene 1
#           gene 5 activated by gene 2
G = 5
basals = [1.5, 4.0, 0.8, 2.0, 1.0]
A = zeros(G, G)
A[1, 3] =  6.0   # gene 3 activates gene 1
A[1, 5] = -5.0   # gene 5 represses gene 1
A[3, 4] =  4.0   # gene 4 activates gene 3
A[4, 1] = -7.0   # gene 1 represses gene 4
A[5, 2] =  5.0   # gene 2 activates gene 5

net = GeneNetwork(basals, A)

# Variable kinetics (non-default)
kin = KineticParams(k_t=3.0, K_d=40.0, n=6.0, mu_m=0.08, mu_p=0.25)

# ── Parameter grid ───────────────────────────────────────────────
cell_nums = [500, 1000, 2000]
Ts = [50.0, 100.0, 250.0]

# Methods to benchmark
methods = [
    ("SSA",              SSA(),                false),
    ("Poisson τ=0.1",    PoissonTauLeap(0.1),  false),
    ("Binomial τ=0.1",   BinomialTauLeap(0.1), false),
    ("Midpoint τ=0.1",   MidpointTauLeap(0.1), false),
    ("CLE dt=0.1",       CLE(0.1),             false),
    ("GPU CLE dt=0.1",   CLE(0.1),             true),
    ("GPU Binom τ=0.1",  BinomialTauLeap(0.1), true),
]

println("="^90)
println("SyntheticscRNAseq.jl Benchmark — G=$G, mixed regulation, variable kinetics")
println("Network: 5 edges (3→1+, 5→1-, 4→3+, 1→4-, 2→5+)")
println("Kinetics: k_t=3.0, K_d=40.0, n=6.0, μ_m=0.08, μ_p=0.25")
println("="^90)

# ── Warmup (compile all methods) ─────────────────────────────────
println("\nWarming up...")
for (name, alg, gpu) in methods
    if gpu
        simulate(net, alg, kin, Val(:gpu); cell_num=100, T=10.0, readout=:protein)
    else
        simulate(net, alg, kin; cell_num=100, T=10.0, readout=:protein, rng=MersenneTwister(1))
    end
end
println("Warmup complete.\n")

# ── Run benchmarks ───────────────────────────────────────────────
# Store results for equivalence comparison
results = Dict{String, Dict{Tuple{Int,Float64}, Tuple{Vector{Float64}, Vector{Float64}, Float64}}}()

for (name, alg, gpu) in methods
    results[name] = Dict()
    println("─"^90)
    @printf("%-20s", name)
    for T in Ts
        @printf("  T=%-5.0f", T)
    end
    println()
    println("─"^90)

    for cell_num in cell_nums
        @printf("  N=%-4d  ", cell_num)
        for T in Ts
            # Skip SSA for large configs (too slow)
            if name == "SSA" && cell_num >= 2000 && T >= 250.0
                @printf("  %8s  ", "skip")
                continue
            end

            GC.gc()
            if gpu
                CUDA.reclaim()
            end

            t_start = time()
            if gpu
                Y = simulate(net, alg, kin, Val(:gpu);
                             cell_num=cell_num, T=T, readout=:protein)
            else
                Y = simulate(net, alg, kin;
                             cell_num=cell_num, T=T, readout=:protein,
                             rng=MersenneTwister(42))
            end
            elapsed = time() - t_start

            means = vec(mean(Y, dims=1))
            vars = vec(var(Y, dims=1))
            results[name][(cell_num, T)] = (means, vars, elapsed)

            @printf("  %6.2fs  ", elapsed)
        end
        println()
    end
end

# ── Equivalence comparison ───────────────────────────────────────
println("\n" * "="^90)
println("EQUIVALENCE CHECK: Mean relative error vs SSA (T=100, N=1000)")
println("="^90)

ref_key = (1000, 100.0)
if haskey(results["SSA"], ref_key)
    ref_means, ref_vars, _ = results["SSA"][ref_key]

    @printf("%-20s  %8s   Gene means                          Gene variances\n", "Method", "Time")
    println("─"^90)

    for (name, _, _) in methods
        if haskey(results[name], ref_key)
            means, vars, elapsed = results[name][ref_key]
            mean_err = mean(abs.(means .- ref_means) ./ max.(ref_means, 1.0))
            var_err = mean(abs.(vars .- ref_vars) ./ max.(ref_vars, 1.0))

            means_str = join([@sprintf("%.1f", m) for m in means], " ")
            vars_str = join([@sprintf("%.0f", v) for v in vars], " ")
            @printf("%-20s  %6.2fs   [%s]   [%s]   mean_err=%.3f var_err=%.3f\n",
                    name, elapsed, means_str, vars_str, mean_err, var_err)
        end
    end

    println("\nSSA reference means: ", [@sprintf("%.1f", m) for m in ref_means])
    println("SSA reference vars:  ", [@sprintf("%.0f", v) for v in ref_vars])
end

# ── Speedup table ────────────────────────────────────────────────
println("\n" * "="^90)
println("SPEEDUP vs SSA")
println("="^90)

@printf("%-20s", "Method")
for (cell_num, T) in [(500, 50.0), (1000, 100.0), (2000, 250.0)]
    @printf("  N=%d,T=%.0f", cell_num, T)
end
println()
println("─"^90)

for (name, _, _) in methods
    @printf("%-20s", name)
    for (cell_num, T) in [(500, 50.0), (1000, 100.0), (2000, 250.0)]
        if haskey(results["SSA"], (cell_num, T)) && haskey(results[name], (cell_num, T))
            ssa_time = results["SSA"][(cell_num, T)][3]
            this_time = results[name][(cell_num, T)][3]
            @printf("  %10.1fx   ", ssa_time / this_time)
        else
            @printf("  %10s   ", "N/A")
        end
    end
    println()
end

println("\nDone!")
