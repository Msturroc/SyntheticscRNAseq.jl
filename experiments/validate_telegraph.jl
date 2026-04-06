#= ================================================================
   Telegraph model validation: exact distributional comparison.

   Compares the empirical mRNA distribution from SSA, BinomialTauLeap
   and CLE against the exact Peccoud-Ycart (1995) / Cao-Grima (2020)
   steady-state distribution across four parameter regimes:

     1. Constitutive limit (fast switching) → Poisson
     2. Moderate bursting → overdispersed unimodal
     3. Strong bursting → geometric-like tail
     4. Slow switching → bimodal (OFF peak + ON peak)

   For each regime, reports:
     - KS statistic (empirical vs exact CDF)
     - Moment errors (mean, variance, Fano factor)
     - L1 distributional distance

   Also generates a 4-panel figure comparing empirical histograms
   against the exact PMF for each regime.

   Usage:
     julia --project=. experiments/validate_telegraph.jl
   ================================================================ =#

using Pkg
Pkg.activate(dirname(@__DIR__))

using SyntheticscRNAseq
using Random
using Statistics
using Printf
using CairoMakie

# ── Parameter regimes ────────────────────────────────────────────
#
# Telegraph model: gene switches ON↔OFF with rates k_on, k_off.
# When ON, transcribes at rate β. mRNA decays at rate μ_m.
#
# Dimensionless: a = k_on/μ_m, b = k_off/μ_m, c = β/μ_m.
# Mean = a*c/(a+b), Fano = 1 + b*c/((a+b)*(a+b+1)).

const MU_M = 0.1     # mRNA decay rate (same across all regimes)
const MU_P = 0.2     # protein decay rate
const K_T  = 1.0     # translation rate

regimes = [
    (name = "Constitutive limit",
     desc = "a=50, b=50, c=40 → fast switching, Poisson-like",
     k_on = 5.0, k_off = 5.0, beta = 4.0),

    (name = "Moderate bursting",
     desc = "a=2, b=5, c=70 → overdispersed unimodal",
     k_on = 0.2, k_off = 0.5, beta = 7.0),

    (name = "Strong bursting",
     desc = "a=0.5, b=5, c=110 → geometric tail",
     k_on = 0.05, k_off = 0.5, beta = 11.0),

    (name = "Bimodal (slow switching)",
     desc = "a=0.2, b=0.3, c=50 → two peaks",
     k_on = 0.02, k_off = 0.03, beta = 5.0),
]

# ── Simulation parameters ────────────────────────────────────────

const N_CELLS = 20000    # large enough for tight KS test
const T_SIM   = 500.0    # sufficient equilibration (slowest timescale ~1/k_on = 50)
const DT      = 0.05     # small dt for tau-leap accuracy

# Note: CLE does not implement telegraph promoter switching — it uses
# the basal rate directly without ON/OFF gating. Only discrete algorithms
# (SSA, tau-leap) handle the telegraph model correctly.
algorithms = [
    ("SSA",             SSA()),
    ("BinomialTauLeap", BinomialTauLeap(DT)),
]

# ── Helper functions ─────────────────────────────────────────────

function empirical_pmf(samples::Vector{Float64}, nmax::Int)
    counts = zeros(nmax + 1)
    for s in samples
        n = round(Int, max(s, 0))
        if n <= nmax
            counts[n + 1] += 1
        end
    end
    return counts ./ length(samples)
end

function l1_distance(p_empirical::Vector{Float64}, p_exact::Vector{Float64})
    nmin = min(length(p_empirical), length(p_exact))
    d = 0.0
    for i in 1:nmin
        d += abs(p_empirical[i] - p_exact[i])
    end
    # Add tail mass from the longer vector
    for i in (nmin+1):length(p_empirical)
        d += abs(p_empirical[i])
    end
    for i in (nmin+1):length(p_exact)
        d += abs(p_exact[i])
    end
    return d
end

function ks_vs_exact(samples::Vector{Float64}, p_exact::Vector{Float64})
    # Build exact CDF
    cdf_exact = cumsum(p_exact)
    n = length(samples)
    sorted = sort(round.(Int, max.(samples, 0)))

    d_max = 0.0
    ecdf_val = 0.0
    for (i, s) in enumerate(sorted)
        ecdf_val = i / n
        idx = min(s + 1, length(cdf_exact))
        exact_cdf = idx > 0 ? cdf_exact[idx] : 0.0
        d_max = max(d_max, abs(ecdf_val - exact_cdf))
        # Also check just before the step
        ecdf_before = (i - 1) / n
        d_max = max(d_max, abs(ecdf_before - exact_cdf))
    end
    return d_max
end

relerr(obs, exact) = abs(obs - exact) / max(abs(exact), 1e-10)

# ═══════════════════════════════════════════════════════════════
#  Run validation
# ═══════════════════════════════════════════════════════════════

println("=" ^ 90)
println("  TELEGRAPH MODEL: Exact Distributional Validation")
println("  Peccoud & Ycart (1995), Cao & Grima (2020)")
println("  N_cells=$N_CELLS, T=$T_SIM, dt=$DT")
println("=" ^ 90)
flush(stdout)

# Storage for plotting
plot_data = []

for (ri, regime) in enumerate(regimes)
    a = regime.k_on / MU_M
    b = regime.k_off / MU_M
    c = regime.beta / MU_M

    println("\n\n", "─" ^ 90)
    println("  Regime $ri: $(regime.name)")
    println("  $(regime.desc)")
    @printf("  a=%.1f, b=%.1f, c=%.1f → mean=%.1f, Fano=%.3f\n",
            a, b, c, telegraph_mean(a, b, c), telegraph_fano(a, b, c))
    println("─" ^ 90)
    flush(stdout)

    # Exact distribution
    p_exact = telegraph_distribution(a, b, c)
    nmax = length(p_exact) - 1
    mean_exact = telegraph_mean(a, b, c)
    var_exact = telegraph_variance(a, b, c)
    fano_exact = telegraph_fano(a, b, c)

    # Create single-gene telegraph network
    net = GeneNetwork([regime.beta], zeros(1, 1);
                      k_on=[regime.k_on], k_off=[regime.k_off])
    kin = KineticParams(k_t=K_T, K_d=50.0, n=2.0, mu_m=MU_M, mu_p=MU_P)

    println()
    @printf("  %-18s | %8s | %8s | %8s | %8s | %8s | %8s\n",
            "Algorithm", "<m> err", "Var err", "Fano err", "KS stat", "L1 dist", "Time")
    println("  ", "-" ^ 82)

    regime_plot = (name=regime.name, a=a, b=b, c=c, p_exact=p_exact, nmax=nmax,
                   empirical=Dict{String,Vector{Float64}}())

    for (alg_name, alg) in algorithms
        rng = MersenneTwister(42)

        t0 = time()
        Y = simulate(net, alg, kin; cell_num=N_CELLS, T=T_SIM,
                     readout=:mrna, rng=rng)
        elapsed = time() - t0

        samples = vec(Y)
        obs_mean = mean(samples)
        obs_var = var(samples)
        obs_fano = obs_var / max(obs_mean, 1e-10)

        p_emp = empirical_pmf(samples, nmax)
        ks = ks_vs_exact(samples, p_exact)
        l1 = l1_distance(p_emp, p_exact)

        @printf("  %-18s | %7.1f%% | %7.1f%% | %7.1f%% | %8.4f | %8.4f | %7.1fs\n",
                alg_name,
                100 * relerr(obs_mean, mean_exact),
                100 * relerr(obs_var, var_exact),
                100 * relerr(obs_fano, fano_exact),
                ks, l1, elapsed)

        regime_plot.empirical[alg_name] = p_emp
        flush(stdout)
    end

    # KS critical value at alpha=0.05
    ks_crit = ks_critical_value(N_CELLS, N_CELLS)
    @printf("\n  KS critical value (α=0.05, N=%d): %.4f\n", N_CELLS, ks_crit)

    push!(plot_data, regime_plot)
end

# ═══════════════════════════════════════════════════════════════
#  Generate figure
# ═══════════════════════════════════════════════════════════════

println("\n\n", "=" ^ 90)
println("  Generating figure...")
println("=" ^ 90)

fig = Figure(size=(1200, 900))

colors = Dict(
    "SSA" => :steelblue,
    "BinomialTauLeap" => :darkorange,
)

for (i, rd) in enumerate(plot_data)
    row = (i - 1) ÷ 2 + 1
    col = (i - 1) % 2 + 1

    ax = Axis(fig[row, col];
              title="$(rd.name)\n(a=$(round(rd.a, digits=1)), b=$(round(rd.b, digits=1)), c=$(round(rd.c, digits=1)))",
              xlabel="mRNA count n",
              ylabel="P(n)",
              titlesize=13)

    # Exact PMF as grey bars
    ns = 0:(length(rd.p_exact)-1)
    barplot!(ax, collect(ns), rd.p_exact;
             color=(:grey70, 0.5), strokewidth=0,
             label="Exact (Peccoud-Ycart)")

    # Overlay empirical PMFs as step lines
    for (alg_name, p_emp) in rd.empirical
        ns_emp = 0:(length(p_emp)-1)
        lines!(ax, collect(ns_emp), p_emp;
               color=colors[alg_name], linewidth=1.5,
               label=alg_name)
    end

    # Truncate x-axis to visible range
    μ = telegraph_mean(rd.a, rd.b, rd.c)
    σ = sqrt(telegraph_variance(rd.a, rd.b, rd.c))
    xlims!(ax, (-1, min(ceil(Int, μ + 5σ), rd.nmax)))
end

# Legend
Legend(fig[3, 1:2],
       [PolyElement(color=(:grey70, 0.5)),
        LineElement(color=:steelblue, linewidth=2),
        LineElement(color=:darkorange, linewidth=2)],
       ["Exact (Peccoud-Ycart)", "SSA", "BinomialTauLeap"];
       orientation=:horizontal, tellheight=true, tellwidth=false,
       framevisible=false)

figpath = joinpath(dirname(@__DIR__), "figures", "telegraph_validation.png")
save(figpath, fig; px_per_unit=3)
println("  Saved: $figpath")

# ═══════════════════════════════════════════════════���═══════════
#  CLE accuracy scaling (Grima 2011)
# ═══════════════════════════════════════════════════════════════

println("\n\n", "=" ^ 90)
println("  GRIMA (2011): CLE Accuracy Scaling with System Size")
println("  Constitutive gene (no telegraph): CLE applicable")
println("  Errors vs exact analytical moments, swept by beta")
println("=" ^ 90)
flush(stdout)

# Use constitutive genes (no telegraph switching) so CLE is applicable.
# CLE cannot model discrete promoter states -- that's shown above.
# Here we test Grima's result: CLE accuracy improves with system size.

beta_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
const N_SEEDS = 10  # average over multiple seeds to reduce sampling noise

println()
@printf("  %-6s | %8s | %12s | %12s | %14s | %14s\n",
        "beta", "<m>exact", "BinTau <m>err", "CLE <m>err", "BinTau F(p)err", "CLE F(p)err")
println("  ", "-" ^ 78)

omega_values = Float64[]
cle_mean_errors = Float64[]
cle_fano_errors = Float64[]
bt_mean_errors = Float64[]
bt_fano_errors = Float64[]

# Exact constitutive moments: <m> = beta/mu_m (Poisson), F(p) = 1 + k_t/(mu_m + mu_p)
fano_p_exact = 1.0 + K_T / (MU_M + MU_P)

for beta_g in beta_values
    mean_m_exact = beta_g / MU_M

    # Constitutive gene (no telegraph)
    net_g = GeneNetwork([beta_g], zeros(1, 1))
    kin_g = KineticParams(k_t=K_T, K_d=50.0, n=2.0, mu_m=MU_M, mu_p=MU_P)

    # Average over N_SEEDS independent runs
    bt_m_means = Float64[]
    bt_p_fanos = Float64[]
    cle_m_means = Float64[]
    cle_p_fanos = Float64[]

    for seed in 1:N_SEEDS
        rng = MersenneTwister(41 + seed)
        Y_bt = simulate(net_g, BinomialTauLeap(DT), kin_g;
                        cell_num=N_CELLS, T=T_SIM, readout=:both, rng=rng)
        push!(bt_m_means, mean(Y_bt[:, 1]))
        push!(bt_p_fanos, var(Y_bt[:, 2]) / max(mean(Y_bt[:, 2]), 1e-10))

        rng = MersenneTwister(41 + seed)
        Y_cle = simulate(net_g, CLE(DT), kin_g;
                         cell_num=N_CELLS, T=T_SIM, readout=:both, rng=rng)
        push!(cle_m_means, mean(Y_cle[:, 1]))
        push!(cle_p_fanos, var(Y_cle[:, 2]) / max(mean(Y_cle[:, 2]), 1e-10))
    end

    bt_m_mean = mean(bt_m_means)
    bt_p_fano = mean(bt_p_fanos)
    cle_m_mean = mean(cle_m_means)
    cle_p_fano = mean(cle_p_fanos)

    # Compare against exact analytical values
    bt_m_err = relerr(bt_m_mean, mean_m_exact)
    cle_m_err = relerr(cle_m_mean, mean_m_exact)
    bt_f_err = relerr(bt_p_fano, fano_p_exact)
    cle_f_err = relerr(cle_p_fano, fano_p_exact)

    @printf("  %-6.2f | %8.1f | %10.2f%%  | %10.2f%%  | %12.2f%%  | %12.2f%%\n",
            beta_g, mean_m_exact, 100*bt_m_err, 100*cle_m_err, 100*bt_f_err, 100*cle_f_err)
    flush(stdout)

    push!(omega_values, mean_m_exact)
    push!(cle_mean_errors, max(cle_m_err, 1e-6))
    push!(cle_fano_errors, max(cle_f_err, 1e-6))
    push!(bt_mean_errors, max(bt_m_err, 1e-6))
    push!(bt_fano_errors, max(bt_f_err, 1e-6))
end

# Fit convergence rates
if length(omega_values) >= 3
    cle_mean_rate = convergence_rate(omega_values, cle_mean_errors)
    cle_fano_rate = convergence_rate(omega_values, cle_fano_errors)
    bt_mean_rate = convergence_rate(omega_values, bt_mean_errors)
    bt_fano_rate = convergence_rate(omega_values, bt_fano_errors)

    println()
    @printf("  CLE  mean error scaling: Ω^{%.2f}  (Grima predicts Ω^{-1.5})\n", cle_mean_rate)
    @printf("  CLE  Fano error scaling: Ω^{%.2f}  (Grima predicts Ω^{-2.0})\n", cle_fano_rate)
    @printf("  BinTau mean error scaling: Ω^{%.2f}\n", bt_mean_rate)
    @printf("  BinTau Fano error scaling: Ω^{%.2f}\n", bt_fano_rate)
end

# ═══════════════════════════════════════════════════════════════
#  CLE accuracy scaling figure
# ═══════════════════════════════════════════════════════════════

fig2 = Figure(size=(700, 400))

ax_mean = Axis(fig2[1, 1];
               title="mRNA mean error vs system size\n(constitutive gene)",
               xlabel="mean mRNA count",
               ylabel="Relative error",
               xscale=log10, yscale=log10,
               titlesize=13)
ax_fano = Axis(fig2[1, 2];
               title="Protein Fano error vs system size\n(constitutive gene, vs exact)",
               xlabel="mean mRNA count",
               ylabel="Relative error (vs exact)",
               xscale=log10, yscale=log10,
               titlesize=13)

scatter!(ax_mean, omega_values, cle_mean_errors; color=:forestgreen, markersize=10, label="CLE")
scatter!(ax_mean, omega_values, bt_mean_errors; color=:darkorange, markersize=10, label="BinomialTauLeap")

# Reference slopes anchored to CLE data (fit intercept, use Grima's slope)
ω_ref = range(minimum(omega_values), maximum(omega_values), length=50)
# Anchor at geometric mean of CLE data
i_mid = length(cle_mean_errors) ÷ 2
c_mean = cle_mean_errors[i_mid] * omega_values[i_mid]^1.5
lines!(ax_mean, collect(ω_ref), c_mean .* collect(ω_ref) .^ (-1.5);
       color=:grey50, linestyle=:dash, linewidth=1.5,
       label="slope -3/2 (Grima)")

scatter!(ax_fano, omega_values, cle_fano_errors; color=:forestgreen, markersize=10, label="CLE")
scatter!(ax_fano, omega_values, bt_fano_errors; color=:darkorange, markersize=10, label="BinomialTauLeap")
c_fano = cle_fano_errors[i_mid] * omega_values[i_mid]^2.0
lines!(ax_fano, collect(ω_ref), c_fano .* collect(ω_ref) .^ (-2.0);
       color=:grey50, linestyle=:dash, linewidth=1.5,
       label="slope -2 (Grima)")

axislegend(ax_mean; position=:rt, framevisible=false, labelsize=10)
axislegend(ax_fano; position=:rt, framevisible=false, labelsize=10)

figpath2 = joinpath(dirname(@__DIR__), "figures", "cle_accuracy_scaling.png")
save(figpath2, fig2; px_per_unit=3)
println("\n  Saved: $figpath2")

println("\nDone.")
