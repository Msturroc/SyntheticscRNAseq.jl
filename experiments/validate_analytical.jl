#= ================================================================
   Analytical validation of all simulation algorithms.

   Sweeps all 7 CPU methods (+ GPU if available) over a single
   constitutive gene with known parameters and compares to exact
   two-stage gene expression formulas (Thattai & van Oudenaarden
   2001). Also tests multi-gene networks.

   Output: formatted error table showing relative errors and timing
   for each method, enabling default algorithm selection.

   Usage:
     julia --project=. experiments/validate_analytical.jl
   ================================================================ =#

using Pkg
Pkg.activate(dirname(@__DIR__))

using SyntheticscRNAseq
using Random
using Statistics
using Printf

# Try to load CUDA for GPU benchmarks
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end
HAS_CUDA && println("CUDA available: ", CUDA.name(CUDA.device()))

# ── Analytical formulas (no dilution, no regulation) ────────────
# Two-stage model: ∅ →β mRNA →μ_m ∅; mRNA →k_t protein →μ_p ∅
#
# Exact moments at steady state:
#   <m>      = β / μ_m
#   Var(m)   = β / μ_m  (Poisson)
#   <p>      = k_t * β / (μ_m * μ_p)
#   Var(p)   = <p> * (1 + k_t / (μ_m + μ_p))
#   Fano(p)  = 1 + k_t / (μ_m + μ_p)
#   Cov(p,m) = k_t * <m> / (μ_m + μ_p)

function analytical_moments(beta, mu_m, mu_p, k_t)
    m_mean = beta / mu_m
    m_var  = m_mean  # Poisson
    p_mean = k_t * beta / (mu_m * mu_p)
    p_fano = 1.0 + k_t / (mu_m + mu_p)
    p_var  = p_mean * p_fano
    cov_pm = k_t * m_mean / (mu_m + mu_p)
    return (; m_mean, m_var, p_mean, p_var, p_fano, cov_pm)
end

# Sturrock & Sturrock 2026: two-stage with growth-coupled dilution μ
function analytical_moments_dilution(beta, mu_m, mu_p, k_t, mu)
    m_mean = beta / (mu_m + mu)
    m_var  = m_mean  # still Poisson (birth-death with effective rate)
    p_mean = k_t * m_mean / (mu_p + mu)
    p_fano = 1.0 + k_t / (mu_m + mu_p + 2*mu)
    p_var  = p_mean * p_fano
    cov_pm = k_t * m_mean / (mu_m + mu_p + 2*mu)
    return (; m_mean, m_var, p_mean, p_var, p_fano, cov_pm)
end

function compute_moments(Y_both)
    m = Y_both[:, 1]
    p = Y_both[:, 2]
    return (
        m_mean = mean(m),
        m_var  = var(m),
        p_mean = mean(p),
        p_var  = var(p),
        p_fano = var(p) / mean(p),
        cov_pm = cov(p, m),
    )
end

relerr(obs, exact) = abs(obs - exact) / max(abs(exact), 1e-10)

# ═════════���══════════════════════════════════��══════════════════
#  1. Single constitutive gene — exact validation
# ═══════════════════════════════════════════════════════════════

println("=" ^ 80)
println("  ANALYTICAL VALIDATION: Single Constitutive Gene (G=1)")
println("=" ^ 80)

beta = 2.0
mu_m = 0.1
mu_p = 0.2
k_t = 1.0

net = GeneNetwork(1, [beta], zeros(1, 1))
kin = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)
exact = analytical_moments(beta, mu_m, mu_p, k_t)

println("\nExact moments:")
@printf("  <m>    = %.3f\n", exact.m_mean)
@printf("  Var(m) = %.3f\n", exact.m_var)
@printf("  <p>    = %.3f\n", exact.p_mean)
@printf("  Var(p) = %.3f\n", exact.p_var)
@printf("  F(p)   = %.3f\n", exact.p_fano)
@printf("  Cov(p,m) = %.3f\n", exact.cov_pm)

T_sim = 500.0
N_cells = 20000

# Algorithm list: (name, algorithm, dt, gpu)
algorithms = [
    ("SSA",                 SSA(),                  nothing, false),
    ("PoissonTauLeap",      PoissonTauLeap(0.1),    0.1,    false),
    ("BinomialTauLeap",     BinomialTauLeap(0.1),   0.1,    false),
    ("MidpointTauLeap",     MidpointTauLeap(0.1),   0.1,    false),
    ("CLE",                 CLE(0.1),               0.1,    false),
    ("CLEFast",             CLEFast(0.1),            0.1,    false),
    ("BinomialTauLeapFast", BinomialTauLeapFast(0.1), 0.1,  false),
]
if HAS_CUDA
    push!(algorithms, ("GPU CLE",             CLE(0.1),             0.1, true))
    push!(algorithms, ("GPU BinomialTauLeap", BinomialTauLeap(0.1), 0.1, true))
end

# Header
println("\n", "-" ^ 100)
@printf("%-22s | %8s | %10s | %8s | %8s | %10s | %8s\n",
        "Method", "<m> err", "Var(m) err", "<p> err", "F(p) err", "Cov(p,m) err", "Time")
println("-" ^ 100)

results = []

for (name, alg, _dt, gpu) in algorithms
    rng = MersenneTwister(42)

    # Warmup (small run to trigger compilation)
    try
        if gpu
            simulate(net, alg, kin, Val(:gpu); cell_num=10, T=10.0, readout=:both)
        else
            simulate(net, alg, kin; cell_num=10, T=10.0, readout=:both, rng=MersenneTwister(1))
        end
    catch e
        println("  WARN: $name warmup failed: $e")
        continue
    end

    # Timed run
    t_start = time()
    if gpu
        Y = simulate(net, alg, kin, Val(:gpu); cell_num=N_cells, T=T_sim, readout=:both)
    else
        Y = simulate(net, alg, kin; cell_num=N_cells, T=T_sim, readout=:both, rng=rng)
    end
    elapsed = time() - t_start

    mom = compute_moments(Y)

    errs = (
        m_mean = relerr(mom.m_mean, exact.m_mean),
        m_var  = relerr(mom.m_var, exact.m_var),
        p_mean = relerr(mom.p_mean, exact.p_mean),
        p_fano = relerr(mom.p_fano, exact.p_fano),
        cov_pm = relerr(mom.cov_pm, exact.cov_pm),
    )

    @printf("%-22s | %7.1f%% | %9.1f%% | %7.1f%% | %7.1f%% | %9.1f%%  | %7.2fs\n",
            name,
            100*errs.m_mean, 100*errs.m_var,
            100*errs.p_mean, 100*errs.p_fano, 100*errs.cov_pm,
            elapsed)

    push!(results, (name=name, errs=errs, time=elapsed, moments=mom))
end

println("-" ^ 100)

# ═══════════════════════════════════════════════════════════════
#  2. Five-gene unregulated — method agreement
# ═══════════════════════════════════════════════════════════════

println("\n\n", "=" ^ 80)
println("  METHOD AGREEMENT: Five Independent Genes (G=5, no regulation)")
println("=" ^ 80)

betas5 = [1.0, 2.0, 3.0, 4.0, 5.0]
net5 = GeneNetwork(5, betas5, zeros(5, 5))
kin5 = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)

T5 = 300.0
N5 = 5000

println("\nExpected protein means: ", [k_t * b / (mu_m * mu_p) for b in betas5])

# Use SSA as reference
rng_ref = MersenneTwister(42)
Y_ref = simulate(net5, SSA(), kin5; cell_num=N5, T=T5, readout=:protein, rng=rng_ref)
ref_means = vec(mean(Y_ref, dims=1))
ref_vars = vec(var(Y_ref, dims=1))

println("\nSSA reference means: ", round.(ref_means, digits=1))
println("SSA reference vars:  ", round.(ref_vars, digits=1))

println("\n", "-" ^ 80)
@printf("%-22s | %12s | %12s | %8s\n",
        "Method", "Mean max err", "Var max err", "Time")
println("-" ^ 80)

for (name, alg, _dt, gpu) in algorithms
    name == "SSA" && continue

    # Warmup
    if gpu
        simulate(net5, alg, kin5, Val(:gpu); cell_num=100, T=10.0, readout=:protein)
    end

    rng = MersenneTwister(42)
    t_start = time()
    if gpu
        Y = simulate(net5, alg, kin5, Val(:gpu); cell_num=N5, T=T5, readout=:protein)
    else
        Y = simulate(net5, alg, kin5; cell_num=N5, T=T5, readout=:protein, rng=rng)
    end
    elapsed = time() - t_start

    sim_means = vec(mean(Y, dims=1))
    sim_vars = vec(var(Y, dims=1))

    mean_errs = [relerr(sim_means[g], ref_means[g]) for g in 1:5]
    var_errs = [relerr(sim_vars[g], ref_vars[g]) for g in 1:5]

    @printf("%-22s | %10.1f%%  | %10.1f%%  | %7.2fs\n",
            name, 100*maximum(mean_errs), 100*maximum(var_errs), elapsed)
end

println("-" ^ 80)

# ═════════════════════════════════════════���═════════════════════
#  3. Five-gene WITH regulation — method agreement vs SSA
# ═══════════════════════════════════════════════════════════════

println("\n\n", "=" ^ 80)
println("  METHOD AGREEMENT: Five-Gene Regulated Network (G=5)")
println("=" ^ 80)

betas_reg = [2.0, 2.0, 2.0, 2.0, 2.0]
A_reg = zeros(5, 5)
A_reg[1, 2] = 5.0    # gene 2 protein activates gene 1
A_reg[2, 1] = -5.0   # gene 1 protein represses gene 2
A_reg[3, 4] = 4.0    # gene 4 activates gene 3
A_reg[5, 3] = -3.0   # gene 3 represses gene 5

net_reg = GeneNetwork(5, betas_reg, A_reg)

# SSA reference (smaller sample due to cost)
N_reg = 3000
T_reg = 200.0

rng_ref = MersenneTwister(42)
Y_ref_reg = simulate(net_reg, SSA(), kin5; cell_num=N_reg, T=T_reg,
                     readout=:protein, rng=rng_ref)
ref_means_reg = vec(mean(Y_ref_reg, dims=1))
ref_vars_reg = vec(var(Y_ref_reg, dims=1))

println("\nSSA reference means: ", round.(ref_means_reg, digits=1))

println("\n", "-" ^ 80)
@printf("%-22s | %12s | %12s | %8s\n",
        "Method", "Mean max err", "Var max err", "Time")
println("-" ^ 80)

for (name, alg, _dt, gpu) in algorithms
    name == "SSA" && continue

    rng = MersenneTwister(42)
    t_start = time()
    if gpu
        Y = simulate(net_reg, alg, kin5, Val(:gpu); cell_num=N_reg, T=T_reg,
                     readout=:protein)
    else
        Y = simulate(net_reg, alg, kin5; cell_num=N_reg, T=T_reg,
                     readout=:protein, rng=rng)
    end
    elapsed = time() - t_start

    sim_means = vec(mean(Y, dims=1))
    sim_vars = vec(var(Y, dims=1))

    mean_errs = [relerr(sim_means[g], ref_means_reg[g]) for g in 1:5]
    var_errs = [relerr(sim_vars[g], ref_vars_reg[g]) for g in 1:5]

    @printf("%-22s | %10.1f%%  | %10.1f%%  | %7.2fs\n",
            name, 100*maximum(mean_errs), 100*maximum(var_errs), elapsed)
end

println("-" ^ 80)

# ═══════════════════════════════════════════════════════════════
#  3.5. Dilution rate — Sturrock & Sturrock 2026 formulas
# ═══════════════════════════════════════════════════════════════

println("\n\n", "=" ^ 80)
println("  DILUTION RATE: Sturrock & Sturrock 2026")
println("  <m> = β/(μ_m + μ), F(p) = 1 + k_t/(μ_m + μ_p + 2μ)")
println("=" ^ 80)

mu_dilution = 0.03  # growth-coupled dilution
kin_dil = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p,
                        dilution=mu_dilution)

exact_dil = analytical_moments_dilution(beta, mu_m, mu_p, k_t, mu_dilution)

println("\nExact moments with dilution μ=$(mu_dilution):")
@printf("  <m>    = %.3f  (vs %.3f without dilution)\n", exact_dil.m_mean, exact.m_mean)
@printf("  <p>    = %.3f  (vs %.3f without dilution)\n", exact_dil.p_mean, exact.p_mean)
@printf("  F(p)   = %.3f  (vs %.3f without dilution)\n", exact_dil.p_fano, exact.p_fano)

println("\n", "-" ^ 100)
@printf("%-22s | %8s | %10s | %8s | %8s | %8s\n",
        "Method", "<m> err", "Var(m) err", "<p> err", "F(p) err", "Time")
println("-" ^ 100)

for (name, alg, _dt, gpu) in algorithms
    rng = MersenneTwister(42)
    t_start = time()
    if gpu
        Y = simulate(net, alg, kin_dil, Val(:gpu); cell_num=N_cells, T=T_sim,
                     readout=:both)
    else
        Y = simulate(net, alg, kin_dil; cell_num=N_cells, T=T_sim,
                     readout=:both, rng=rng)
    end
    elapsed = time() - t_start

    mom = compute_moments(Y)
    errs = (
        m_mean = relerr(mom.m_mean, exact_dil.m_mean),
        m_var  = relerr(mom.m_var, exact_dil.m_var),
        p_mean = relerr(mom.p_mean, exact_dil.p_mean),
        p_fano = relerr(mom.p_fano, exact_dil.p_fano),
    )

    @printf("%-22s | %7.1f%% | %9.1f%% | %7.1f%% | %7.1f%% | %7.2fs\n",
            name, 100*errs.m_mean, 100*errs.m_var,
            100*errs.p_mean, 100*errs.p_fano, elapsed)
end

println("-" ^ 100)

# ═══════════════════════════════════════════════════════════════
#  4. Population dynamics — Moran process validation
# ═══════════════════════════════════════════════════════════════

println("\n\n", "=" ^ 80)
println("  POPULATION DYNAMICS: Moran Process Validation")
println("=" ^ 80)

pop = PopulationConfig(cell_num=2000, growth_rate=0.03,
                       V_div=2.0, V_init=(0.8, 1.2),
                       div_check_interval=5)

# BinomialTauLeap with population
rng = MersenneTwister(42)
t_start = time()
Y_pop_bt = simulate(net, BinomialTauLeap(0.1), kin;
                    cell_num=2000, T=200.0, readout=:both,
                    rng=rng, population=pop)
elapsed_bt = time() - t_start

# CLE with population
rng = MersenneTwister(42)
t_start = time()
Y_pop_cle = simulate(net, CLE(0.1), kin;
                     cell_num=2000, T=200.0, readout=:both,
                     rng=rng, population=pop)
elapsed_cle = time() - t_start

# Also run with dilution=growth_rate (no population) for comparison
rng = MersenneTwister(42)
t_start = time()
Y_dil = simulate(net, BinomialTauLeap(0.1), kin_dil;
                 cell_num=2000, T=200.0, readout=:both, rng=rng)
elapsed_dil = time() - t_start

println("\nComparing population (Moran + volume-dep) vs dilution (constant μ):")
println("Dilution-mode exact: <m> = $(round(exact_dil.m_mean, digits=2)), <p> = $(round(exact_dil.p_mean, digits=2)), F(p) = $(round(exact_dil.p_fano, digits=3))")
println()

@printf("%-28s | %8s | %8s | %10s | %8s\n",
        "Method", "<m>", "<p>", "F(p)", "Time")
println("-" ^ 72)

m_bt = mean(Y_pop_bt[:, 1])
p_bt = mean(Y_pop_bt[:, 2])
f_bt = var(Y_pop_bt[:, 2]) / mean(Y_pop_bt[:, 2])
@printf("%-28s | %8.1f | %8.1f | %10.2f | %7.2fs\n",
        "BinomialTauLeap + population", m_bt, p_bt, f_bt, elapsed_bt)

m_cle = mean(Y_pop_cle[:, 1])
p_cle = mean(Y_pop_cle[:, 2])
f_cle = var(Y_pop_cle[:, 2]) / mean(Y_pop_cle[:, 2])
@printf("%-28s | %8.1f | %8.1f | %10.2f | %7.2fs\n",
        "CLE + population", m_cle, p_cle, f_cle, elapsed_cle)

m_dil = mean(Y_dil[:, 1])
p_dil = mean(Y_dil[:, 2])
f_dil = var(Y_dil[:, 2]) / mean(Y_dil[:, 2])
@printf("%-28s | %8.1f | %8.1f | %10.2f | %7.2fs\n",
        "BinomialTauLeap + dilution", m_dil, p_dil, f_dil, elapsed_dil)

@printf("%-28s | %8.1f | %8.1f | %10.3f |\n",
        "Exact (with dilution)", exact_dil.m_mean, exact_dil.p_mean, exact_dil.p_fano)

println("-" ^ 72)
println("\nNote: population model uses volume-dependent transcription → means scale with <V>.")
println("Dilution model uses constant-rate dilution → matches exact Sturrock formulas.")
println("Cell count preserved: BinomialTauLeap=$(size(Y_pop_bt, 1)), CLE=$(size(Y_pop_cle, 1))")

# ═════════════════════════════════════════════════���═════════════
#  5. Fano factor sweep — verify linear relationship with k_t
# ═══════════════════════════════���═══════════════════════════════

println("\n\n", "=" ^ 80)
println("  FANO FACTOR SWEEP: F(p) = 1 + k_t/(μ_m + μ_p)")
println("=" ^ 80)

k_t_values = [0.5, 1.0, 2.0, 4.0, 8.0]

println("\n", "-" ^ 70)
@printf("%-6s | %8s | %8s | %8s | %8s\n",
        "k_t", "F exact", "F SSA", "F BinTau", "F CLE")
println("-" ^ 70)

for kt in k_t_values
    kin_kt = KineticParams(k_t=kt, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)
    fano_exact = 1.0 + kt / (mu_m + mu_p)

    rng = MersenneTwister(42)
    Y_ssa = simulate(net, SSA(), kin_kt; cell_num=15000, T=500.0,
                     readout=:protein, rng=rng)
    fano_ssa = var(Y_ssa[:, 1]) / mean(Y_ssa[:, 1])

    rng = MersenneTwister(42)
    Y_bt = simulate(net, BinomialTauLeap(0.05), kin_kt; cell_num=15000, T=500.0,
                    readout=:protein, rng=rng)
    fano_bt = var(Y_bt[:, 1]) / mean(Y_bt[:, 1])

    rng = MersenneTwister(42)
    Y_cle = simulate(net, CLE(0.05), kin_kt; cell_num=15000, T=500.0,
                     readout=:protein, rng=rng)
    fano_cle = var(Y_cle[:, 1]) / mean(Y_cle[:, 1])

    @printf("%-6.1f | %8.3f | %8.3f | %8.3f | %8.3f\n",
            kt, fano_exact, fano_ssa, fano_bt, fano_cle)
end

println("-" ^ 70)

# ════════════════════════════════���══════════════════════════════
#  6. Summary: accuracy ranking + recommended default
# ═══════════════════════════════════════���═══════════════════════

println("\n\n", "=" ^ 80)
println("  SUMMARY: Accuracy Ranking (G=1 single gene)")
println("=" ^ 80)

# Compute max error across all moments for each method
println("\n", "-" ^ 65)
@printf("%-22s | %10s | %10s | %8s\n",
        "Method", "Max err", "Mean err", "Time")
println("-" ^ 65)

accuracy_threshold = 0.10  # 10% max error

for r in results
    max_err = maximum([r.errs.m_mean, r.errs.m_var, r.errs.p_mean,
                       r.errs.p_fano, r.errs.cov_pm])
    mean_err = mean([r.errs.m_mean, r.errs.m_var, r.errs.p_mean,
                     r.errs.p_fano, r.errs.cov_pm])
    pass = max_err < accuracy_threshold ? "PASS" : "FAIL"
    @printf("%-22s | %8.1f%%  | %8.1f%%  | %7.2fs  %s\n",
            r.name, 100*max_err, 100*mean_err, r.time, pass)
end

println("-" ^ 65)

# Find fastest method that passes
passing = filter(r -> begin
    max_err = maximum([r.errs.m_mean, r.errs.m_var, r.errs.p_mean,
                       r.errs.p_fano, r.errs.cov_pm])
    max_err < accuracy_threshold
end, results)

if !isempty(passing)
    best = sort(passing, by=r -> r.time)[1]
    println("\nRecommended default: $(best.name)")
    @printf("  Max error: %.1f%%, Time: %.2fs\n",
            100*maximum([best.errs.m_mean, best.errs.m_var, best.errs.p_mean,
                         best.errs.p_fano, best.errs.cov_pm]),
            best.time)
else
    println("\nWARNING: No method passed the $(100*accuracy_threshold)% accuracy threshold.")
    println("Consider using smaller dt or more cells.")
end

# ═══════════════════════════════════════════════════════════════
#  7. Thomas — population volume distribution
# ═══════════════════════════════════════════════════════════════

println("\n\n", "=" ^ 80)
println("  THOMAS: Population Volume Distribution")
println("  V ∈ [V_div/2, V_div], <V> = V_div/(2ln2), <1/V> = 2ln2/V_div")
println("=" ^ 80)

V_div_val = pop.V_div
rng = MersenneTwister(42)
Y_popv, state_v = simulate_with_state(net, BinomialTauLeap(0.1), kin, pop;
                                       T=500.0, readout=:both, rng=rng)
V_snap = state_v.volumes

V_mean_exact = V_div_val / (2 * log(2))
inv_V_exact = 2 * log(2) / V_div_val
V_median_exact = V_div_val / sqrt(2)

println("\n  Volume statistics (N=$(length(V_snap)) cells):")
@printf("    <V>   observed: %.4f   exact: %.4f   err: %.1f%%\n",
        mean(V_snap), V_mean_exact, 100*relerr(mean(V_snap), V_mean_exact))
@printf("    <1/V> observed: %.4f   exact: %.4f   err: %.1f%%\n",
        mean(1.0 ./ V_snap), inv_V_exact, 100*relerr(mean(1.0 ./ V_snap), inv_V_exact))
@printf("    med   observed: %.4f   exact: %.4f   err: %.1f%%\n",
        median(V_snap), V_median_exact, 100*relerr(median(V_snap), V_median_exact))
@printf("    min: %.3f  max: %.3f  (bounds: [%.1f, %.1f])\n",
        minimum(V_snap), maximum(V_snap), V_div_val/2, V_div_val)

# mRNA-volume correlation (Thomas snapshot bias)
m_snap = Y_popv[:, 1]
corr_mV = cor(m_snap, V_snap)
@printf("\n    mRNA-volume correlation: %.3f (expect > 0 from vol-dep transcription)\n", corr_mV)

# ═══════════════════════════════════════════════════════════════
#  8. Sturrock — population ↔ dilution concentration equivalence
# ═══════════════════════════════════════════════════════════════

println("\n\n", "=" ^ 80)
println("  STURROCK: Population ↔ Dilution Equivalence (Appendix E)")
println("  Concentration c = m/V should match dilution model mean")
println("=" ^ 80)

# Population model: volume-dependent transcription, no dilution term
kin_nodil_sturrock = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)
kin_dil_sturrock = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p,
                                 dilution=pop.growth_rate)

rng = MersenneTwister(42)
Y_dil_s = simulate(net, BinomialTauLeap(0.1), kin_dil_sturrock;
                   cell_num=3000, T=500.0, readout=:both, rng=rng)

rng = MersenneTwister(42)
Y_pop_s, state_s = simulate_with_state(net, BinomialTauLeap(0.1), kin_nodil_sturrock, pop;
                                        T=500.0, readout=:both, rng=rng)
V_s = state_s.volumes

c_m_exact = beta / (mu_m + pop.growth_rate)
c_p_exact = k_t * c_m_exact / (mu_p + pop.growth_rate)

c_m_pop = mean(Y_pop_s[:, 1] ./ V_s)
c_p_pop = mean(Y_pop_s[:, 2] ./ V_s)
m_dil = mean(Y_dil_s[:, 1])
p_dil = mean(Y_dil_s[:, 2])

println("\n  ", "-" ^ 70)
@printf("  %-25s | %10s | %10s\n", "Quantity", "mRNA", "Protein")
println("  ", "-" ^ 70)
@printf("  %-25s | %10.3f | %10.3f\n", "Exact (β/(μ+λ))", c_m_exact, c_p_exact)
@printf("  %-25s | %10.3f | %10.3f\n", "Dilution model <m>", m_dil, p_dil)
@printf("  %-25s | %10.3f | %10.3f\n", "Population <m/V>", c_m_pop, c_p_pop)
@printf("  %-25s | %9.1f%% | %9.1f%%\n", "Dilution vs exact err",
        100*relerr(m_dil, c_m_exact), 100*relerr(p_dil, c_p_exact))
@printf("  %-25s | %9.1f%% | %9.1f%%\n", "Population vs exact err",
        100*relerr(c_m_pop, c_m_exact), 100*relerr(c_p_pop, c_p_exact))
println("  ", "-" ^ 70)

# No-enrichment check
enrichment = mean(Y_pop_s[:, 1]) / (c_m_pop * mean(V_s))
@printf("\n  No-enrichment ratio <m>/(<c>*<V>): %.3f (should be ≈ 1.0)\n", enrichment)

# ═══════════════════════════════════════════════════════════════
#  9. Grima — LNA breakdown at low molecule counts
# ═══════════════════════════════════════════════════════════════

println("\n\n", "=" ^ 80)
println("  GRIMA: LNA/CLE Breakdown at Low Copy Numbers")
println("  β=0.1 → <m>=1.0 molecule (Poisson)")
println("=" ^ 80)

beta_low = 0.1
net_low = GeneNetwork(1, [beta_low], zeros(1, 1))
m_exact_low = beta_low / mu_m  # = 1.0
p_exact_low = k_t * beta_low / (mu_m * mu_p)  # = 5.0
fano_exact_low = 1.0 + k_t / (mu_m + mu_p)

N_low = 20000
T_low = 500.0

# SSA (exact)
rng = MersenneTwister(42)
Y_ssa_low = simulate(net_low, SSA(), kin; cell_num=N_low, T=T_low,
                     readout=:both, rng=rng)
# BinomialTauLeap
rng = MersenneTwister(42)
Y_bt_low = simulate(net_low, BinomialTauLeap(0.05), kin; cell_num=N_low, T=T_low,
                    readout=:both, rng=rng)
# CLE
rng = MersenneTwister(42)
Y_cle_low = simulate(net_low, CLE(0.05), kin; cell_num=N_low, T=T_low,
                     readout=:both, rng=rng)

println("\n  Exact: <m>=", m_exact_low, ", <p>=", p_exact_low,
        ", F(p)=", round(fano_exact_low, digits=3))

println("\n  ", "-" ^ 85)
@printf("  %-22s | %8s | %8s | %8s | %8s | %8s\n",
        "Method", "<m>", "F(m)", "<p>", "F(p)", "Zeros %")
println("  ", "-" ^ 85)

for (name, Y_low) in [("SSA (exact)", Y_ssa_low),
                       ("BinomialTauLeap", Y_bt_low),
                       ("CLE", Y_cle_low)]
    m_mean_l = mean(Y_low[:, 1])
    m_fano_l = var(Y_low[:, 1]) / max(mean(Y_low[:, 1]), 1e-10)
    p_mean_l = mean(Y_low[:, 2])
    p_fano_l = var(Y_low[:, 2]) / max(mean(Y_low[:, 2]), 1e-10)
    zero_frac = count(Y_low[:, 1] .== 0) / N_low
    @printf("  %-22s | %8.3f | %8.3f | %8.3f | %8.3f | %7.1f%%\n",
            name, m_mean_l, m_fano_l, p_mean_l, p_fano_l, 100*zero_frac)
end
println("  ", "-" ^ 85)

# Fano error comparison
bt_fano_err = relerr(var(Y_bt_low[:, 2]) / mean(Y_bt_low[:, 2]),
                     var(Y_ssa_low[:, 2]) / mean(Y_ssa_low[:, 2]))
cle_fano_err = relerr(var(Y_cle_low[:, 2]) / mean(Y_cle_low[:, 2]),
                      var(Y_ssa_low[:, 2]) / mean(Y_ssa_low[:, 2]))

@printf("\n  Protein Fano error vs SSA:\n")
@printf("    BinomialTauLeap: %.1f%%\n", 100*bt_fano_err)
@printf("    CLE:             %.1f%%\n", 100*cle_fano_err)
if bt_fano_err < cle_fano_err
    println("    → BinomialTauLeap wins (as expected from Grima theory)")
else
    println("    → CLE wins (unexpected — check dt or sample size)")
end

# High-count recovery check
beta_high = 10.0
net_high = GeneNetwork(1, [beta_high], zeros(1, 1))
rng = MersenneTwister(42)
Y_cle_high = simulate(net_high, CLE(0.1), kin; cell_num=10000, T=300.0,
                      readout=:protein, rng=rng)
cle_fano_high = var(Y_cle_high[:, 1]) / mean(Y_cle_high[:, 1])
fano_exact_high = fano_exact_low  # same kinetics, different β

@printf("\n  High-count recovery (β=%.1f, <m>=%.0f):\n", beta_high, beta_high/mu_m)
@printf("    CLE Fano:  %.3f  (exact: %.3f, err: %.1f%%)\n",
        cle_fano_high, fano_exact_high, 100*relerr(cle_fano_high, fano_exact_high))
println("    → CLE recovers at high molecule counts")

println("\nDone.")
