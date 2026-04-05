#= ================================================================
   Generate figures for the README.

   1. Algorithm comparison heatmaps (promoter, mRNA, protein)
   2. Steady-state distributions: mRNA histogram vs Poisson/NB
   3. Population snapshot: mRNA counts coloured by cell volume
   4. Capture model: mean-variance before/after
   5. Real vs synthetic scRNA-seq comparison
   6. Analytical validation

   Usage:
     julia --project=. scripts/generate_figures.jl
   ================================================================ =#

using Pkg
Pkg.activate(dirname(@__DIR__))

println("Loading packages..."); flush(stdout)
using SyntheticscRNAseq
using Random
using Statistics
using CairoMakie
using UMAP
using SparseArrays
println("Packages loaded."); flush(stdout)

mkpath(joinpath(dirname(@__DIR__), "figures"))
figdir = joinpath(dirname(@__DIR__), "figures")

# ═══════════════════════════════════════════════════════════════
#  Figure 1: Algorithm comparison heatmaps
#            3 rows (promoter, mRNA, protein) x 4 cols (SSA, BinomialTL, PoissonTL, CLE)
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 1: algorithm comparison heatmaps..."); flush(stdout)

using Distributions: Binomial as BinomDist, Binomial, Poisson

# ── Trajectory recorders (single-cell, no volume) ──

function ssa_trajectory(; beta=5.0, k_on=0.03, k_off=0.3,
        k_t=2.0, mu_m=0.1, mu_p=0.3, T=1000.0, dt_out=0.1,
        rng=Random.default_rng())
    m = 0; p = 0
    promoter = rand(rng) < k_on / (k_on + k_off)
    n_out = ceil(Int, T / dt_out) + 1
    t_grid = range(0, T, length=n_out)
    ms = zeros(Int, n_out); ps = zeros(Int, n_out); prs = zeros(Int, n_out)
    ms[1] = m; ps[1] = p; prs[1] = Int(promoter)
    out_idx = 2; t = 0.0
    while t < T
        prop_tx = promoter ? beta : 0.0
        prop_md = mu_m * m; prop_tl = k_t * m; prop_pd = mu_p * p
        prop_on = !promoter ? k_on : 0.0; prop_off = promoter ? k_off : 0.0
        a = prop_tx + prop_md + prop_tl + prop_pd + prop_on + prop_off
        a <= 0 && break
        tau = -log(rand(rng)) / a
        t += tau; t > T && break
        while out_idx <= n_out && t_grid[out_idx] <= t
            ms[out_idx] = m; ps[out_idx] = p; prs[out_idx] = Int(promoter)
            out_idx += 1
        end
        r = rand(rng) * a; cum = prop_tx
        if r < cum; m += 1
        else; cum += prop_md
            if r < cum; m = max(m-1, 0)
            else; cum += prop_tl
                if r < cum; p += 1
                else; cum += prop_pd
                    if r < cum; p = max(p-1, 0)
                    else; cum += prop_on
                        if r < cum; promoter = true
                        else; promoter = false; end
                    end; end; end; end
    end
    for i in out_idx:n_out; ms[i] = m; ps[i] = p; prs[i] = Int(promoter); end
    return t_grid, ms, ps, prs
end

function tauleap_trajectory(; beta=5.0, k_on=0.03, k_off=0.3,
        k_t=2.0, mu_m=0.1, mu_p=0.3, T=1000.0, dt=0.1,
        use_binomial=true, rng=Random.default_rng())
    m = 0; p = 0
    promoter = rand(rng) < k_on / (k_on + k_off)
    n_steps = ceil(Int, T / dt)
    ms = zeros(Int, n_steps+1); ps = zeros(Int, n_steps+1); prs = zeros(Int, n_steps+1)
    ms[1] = m; ps[1] = p; prs[1] = Int(promoter)
    for step in 1:n_steps
        if promoter
            if rand(rng) < (1.0 - exp(-k_off * dt)); promoter = false; end
        else
            if rand(rng) < (1.0 - exp(-k_on * dt)); promoter = true; end
        end
        prop_tx = promoter ? beta : 0.0
        n_tx = rand(rng, Poisson(prop_tx * dt))
        if use_binomial && m > 0
            n_md = rand(rng, Binomial(m, 1.0 - exp(-mu_m * dt)))
        else
            n_md = rand(rng, Poisson(mu_m * max(m, 0) * dt))
        end
        m = max(m + n_tx - n_md, 0)
        n_tl = rand(rng, Poisson(k_t * max(m, 0) * dt))
        if use_binomial && p > 0
            n_pd = rand(rng, Binomial(p, 1.0 - exp(-mu_p * dt)))
        else
            n_pd = rand(rng, Poisson(mu_p * max(p, 0) * dt))
        end
        p = max(p + n_tl - n_pd, 0)
        ms[step+1] = m; ps[step+1] = p; prs[step+1] = Int(promoter)
    end
    t_grid = range(0, T, length=n_steps+1)
    return t_grid, ms, ps, prs
end

function cle_trajectory(; beta=5.0, k_on=0.03, k_off=0.3,
        k_t=2.0, mu_m=0.1, mu_p=0.3, T=1000.0, dt=0.1,
        rng=Random.default_rng())
    m = 0.0; p = 0.0
    promoter = rand(rng) < k_on / (k_on + k_off)
    n_steps = ceil(Int, T / dt)
    ms = zeros(Float64, n_steps+1); ps = zeros(Float64, n_steps+1); prs = zeros(Int, n_steps+1)
    ms[1] = m; ps[1] = p; prs[1] = Int(promoter)
    sqrt_dt = sqrt(dt)
    for step in 1:n_steps
        if promoter
            if rand(rng) < (1.0 - exp(-k_off * dt)); promoter = false; end
        else
            if rand(rng) < (1.0 - exp(-k_on * dt)); promoter = true; end
        end
        prop_tx = promoter ? beta : 0.0
        var_m = max(prop_tx + mu_m * max(m, 0.0), 0.0)
        m = max(m + (prop_tx - mu_m * m) * dt + randn(rng) * sqrt_dt * sqrt(var_m), 0.0)
        var_p = max(k_t * max(m, 0.0) + mu_p * max(p, 0.0), 0.0)
        p = max(p + (k_t * m - mu_p * p) * dt + randn(rng) * sqrt_dt * sqrt(var_p), 0.0)
        ms[step+1] = m; ps[step+1] = p; prs[step+1] = Int(promoter)
    end
    t_grid = range(0, T, length=n_steps+1)
    return t_grid, round.(Int, ms), round.(Int, ps), prs
end

# ── Generate heatmap data ──

n_cells = 10; T_fig1 = 1000.0

algs_fig1 = [
    ("SSA", (s) -> ssa_trajectory(; T=T_fig1, dt_out=0.1, rng=MersenneTwister(s))),
    ("BinomialTL", (s) -> tauleap_trajectory(; T=T_fig1, use_binomial=true, rng=MersenneTwister(s))),
    ("PoissonTL", (s) -> tauleap_trajectory(; T=T_fig1, use_binomial=false, rng=MersenneTwister(s))),
    ("CLE", (s) -> cle_trajectory(; T=T_fig1, rng=MersenneTwister(s))),
]

all_data = Dict{String, NamedTuple}()
for (name, fn) in algs_fig1
    println("  $name..."); flush(stdout)
    local t_grid, M, P, PR
    for c in 1:n_cells
        tg, ms, ps, prs = fn(c)
        if c == 1
            t_grid = collect(tg); n_t = length(tg)
            M = zeros(Int, n_t, n_cells); P = zeros(Int, n_t, n_cells); PR = zeros(Int, n_t, n_cells)
        end
        M[:, c] = ms; P[:, c] = ps; PR[:, c] = prs
    end
    all_data[name] = (t=t_grid, M=M, P=P, PR=PR)
end

println("  Plotting heatmaps..."); flush(stdout)
fig1 = Figure(size=(1200, 700))
alg_names = ["SSA", "BinomialTL", "PoissonTL", "CLE"]

for (col, name) in enumerate(alg_names)
    d = all_data[name]

    ax = Axis(fig1[1, col], title=name)
    heatmap!(ax, d.t, 1:n_cells, d.PR, colormap=[:white, :seagreen])
    col > 1 && hideydecorations!(ax, grid=false)
    col == 1 && (ax.ylabel = "Promoter")
    hidexdecorations!(ax, grid=false)

    ax2 = Axis(fig1[2, col])
    heatmap!(ax2, d.t, 1:n_cells, Float64.(d.M), colormap=:Blues)
    col > 1 && hideydecorations!(ax2, grid=false)
    col == 1 && (ax2.ylabel = "mRNA")
    hidexdecorations!(ax2, grid=false)

    ax3 = Axis(fig1[3, col], xlabel="Time")
    heatmap!(ax3, d.t, 1:n_cells, Float64.(d.P), colormap=:Reds)
    col > 1 && hideydecorations!(ax3, grid=false)
    col == 1 && (ax3.ylabel = "Protein")
end

save(joinpath(figdir, "timeseries_bursty.png"), fig1, px_per_unit=3)
println("  Saved timeseries_bursty.png"); flush(stdout)

# ═══════════════════════════════════════════════════════════════
#  Figure 2: mRNA distribution — constitutive vs bursty
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 2: mRNA distributions..."); flush(stdout)

kin_dist = KineticParams(k_t=1.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.2)

# Constitutive
net_c = GeneNetwork(1, [2.0], zeros(1, 1))
rng = MersenneTwister(42)
Y_c = simulate(net_c, SSA(), kin_dist; cell_num=20000, T=500.0, readout=:mrna, rng=rng)
m_c = vec(Y_c)

# Bursty (same mean, higher variance)
net_b = GeneNetwork([10.0], zeros(1, 1); k_on=[0.02], k_off=[0.08])
rng = MersenneTwister(42)
Y_b = simulate(net_b, SSA(), kin_dist; cell_num=20000, T=500.0, readout=:mrna, rng=rng)
m_b = vec(Y_b)

fig2 = Figure(size=(700, 400))
ax = Axis(fig2[1, 1], xlabel="mRNA count", ylabel="Density",
          title="Constitutive vs bursty transcription (same mean)")

hist!(ax, m_c, bins=0:1:maximum(m_c), normalization=:pdf,
      color=(:steelblue, 0.5), label="Constitutive (F=$(round(var(m_c)/mean(m_c), digits=1)))")
hist!(ax, m_b, bins=0:1:min(maximum(m_b), 150), normalization=:pdf,
      color=(:firebrick, 0.5), label="Bursty (F=$(round(var(m_b)/mean(m_b), digits=1)))")

axislegend(ax, position=:rt)
save(joinpath(figdir, "mrna_distribution.png"), fig2, px_per_unit=3)
println("  Saved mrna_distribution.png"); flush(stdout)

# ═══════════════════════════════════════════════════════════════
#  Figure 3: Population dynamics — volume and expression
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 3: population snapshot..."); flush(stdout)

net_pop = GeneNetwork(2, [3.0, 1.5], [0.0 4.0; -3.0 0.0])
kin_pop = KineticParams(k_t=1.0, K_d=50.0, n=4.0, mu_m=0.1, mu_p=0.2)
pop = PopulationConfig(cell_num=2000, growth_rate=0.03,
                       V_div=2.0, V_init=(1.0, 2.0), div_check_interval=1)

rng = MersenneTwister(42)
Y_pop, state = simulate_with_state(net_pop, BinomialTauLeap(0.1), kin_pop, pop;
                                   T=500.0, readout=:both, rng=rng)
V = state.volumes

fig3 = Figure(size=(800, 400))

ax3a = Axis(fig3[1, 1], xlabel="Gene 1 mRNA", ylabel="Gene 2 mRNA",
            title="Population snapshot (colour = cell volume)")
sc = scatter!(ax3a, Y_pop[:, 1], Y_pop[:, 2], color=V,
              colormap=:viridis, markersize=4, alpha=0.6)
Colorbar(fig3[1, 2], sc, label="Volume")

ax3b = Axis(fig3[1, 3], xlabel="Cell volume", ylabel="Count",
            title="Volume distribution")
hist!(ax3b, V, bins=range(pop.V_div/2, pop.V_div, length=30),
      color=(:teal, 0.4), strokecolor=:teal, strokewidth=0.5)

save(joinpath(figdir, "population_snapshot.png"), fig3, px_per_unit=3)
println("  Saved population_snapshot.png"); flush(stdout)

# ═══════════════════════════════════════════════════════════════
#  Figure 4: Capture model — before and after
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 4: capture model effect..."); flush(stdout)

basals_cap = exp.(range(log(0.3), log(10.0), length=20))
net_cap = GeneNetwork(20, basals_cap, zeros(20, 20))
kin_cap = KineticParams(k_t=2.0, K_d=50.0, n=4.0, mu_m=0.1, mu_p=0.2, dilution=0.03)

rng = MersenneTwister(42)
Y_true = simulate(net_cap, BinomialTauLeap(0.1), kin_cap;
                  cell_num=3000, T=300.0, readout=:mrna, rng=rng)

cap10 = CaptureModel(efficiency=0.10, efficiency_std=0.3, readout=:mrna)
Y_cap10 = Float64.(apply_capture(Y_true, cap10; rng=MersenneTwister(42)))

fig4 = Figure(size=(800, 350))

gm_true = vec(mean(Y_true, dims=1))
gv_true = vec(var(Y_true, dims=1))
gm_cap = vec(mean(Y_cap10, dims=1))
gv_cap = vec(var(Y_cap10, dims=1))

ax4a = Axis(fig4[1, 1], xlabel="Mean", ylabel="Variance",
            title="True mRNA counts", xscale=log10, yscale=log10)
scatter!(ax4a, gm_true, gv_true, color=:steelblue, markersize=10)
lines!(ax4a, [1, 100], [1, 100], color=:grey, linestyle=:dash, label="Poisson")

ax4b = Axis(fig4[1, 2], xlabel="Mean", ylabel="Variance",
            title="After capture (10%)", xscale=log10, yscale=log10)
scatter!(ax4b, gm_cap, gv_cap, color=:firebrick, markersize=10)
lines!(ax4b, [0.01, 10], [0.01, 10], color=:grey, linestyle=:dash, label="Poisson")

save(joinpath(figdir, "capture_model.png"), fig4, px_per_unit=3)
println("  Saved capture_model.png"); flush(stdout)

# ═══════════════════════════════════════════════════════════════
#  Figure 5: Real vs synthetic scRNA-seq — per-gene density
#            comparison + mean-variance + Fano scatter
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 5: real vs synthetic comparison..."); flush(stdout)

# Load real monocyte data
function read_10x_mtx(f)
    lines = readlines(f); ds = 1
    while startswith(lines[ds], "%"); ds += 1; end
    h = split(lines[ds])
    ng, nc, nnz_val = parse(Int, h[1]), parse(Int, h[2]), parse(Int, h[3])
    I = Vector{Int}(undef, nnz_val); J = Vector{Int}(undef, nnz_val); V = Vector{Int}(undef, nnz_val)
    for k in 1:nnz_val; p = split(lines[ds+k]); I[k]=parse(Int,p[1]); J[k]=parse(Int,p[2]); V[k]=parse(Int,p[3]); end
    sparse(I, J, V, ng, nc)
end

pbmc_dir = joinpath(dirname(@__DIR__), "experiments", "data", "pbmc3k",
                    "filtered_gene_bc_matrices", "hg19")
if isfile(joinpath(pbmc_dir, "matrix.mtx"))
    X = read_10x_mtx(joinpath(pbmc_dir, "matrix.mtx"))
    gene_names = [split(l, "\t")[2] for l in readlines(joinpath(pbmc_dir, "genes.tsv"))]
    cd14_idx = findfirst(==("CD14"), gene_names)
    lyz_idx = findfirst(==("LYZ"), gene_names)
    mono_cells = findall(c -> X[cd14_idx,c] > 0 || X[lyz_idx,c] > 1, 1:size(X,2))
    X_mono = Matrix{Float64}(X[:, mono_cells])
    n_mono = length(mono_cells)

    # Select top 50 variable genes in monocytes
    gm = vec(mean(X_mono, dims=2))
    gv = vec(var(X_mono, dims=2))
    top50 = sortperm(gv, rev=true)[1:50]
    X_sel = X_mono[top50, :]'  # (n_cells, 50)
    gene_labels = gene_names[top50]

    # ── Per-gene telegraph calibration from observed Fano factors ──
    eff = 0.07           # capture efficiency
    mu_m_cal = 0.1
    dilution_cal = 0.02
    mu_eff = mu_m_cal + dilution_cal  # 0.12
    k_off_ref = 0.5      # fixed reference switching-off rate

    G_fig = 50
    basals_cal  = zeros(G_fig)
    k_on_cal    = fill(Inf, G_fig)
    k_off_cal   = fill(Inf, G_fig)

    for (idx, gi) in enumerate(top50)
        m_obs = gm[gi]
        f_obs = gv[gi] / max(gm[gi], 1e-6)

        m_true = m_obs / eff
        f_true = max(1.0 + (f_obs - 1.0) / eff, 1.1)

        if f_obs < 1.1
            basals_cal[idx] = m_true * mu_eff
        else
            burst_size = max(f_true - 1.0, 0.5)
            basals_cal[idx] = burst_size * k_off_ref
            k_on_cal[idx] = clamp(m_true * mu_eff / burst_size, 0.001, 100.0)
            k_off_cal[idx] = k_off_ref
        end
    end

    # ── Sparse regulation: 4 TFs, each regulating 3-5 targets ──
    W_fig = zeros(G_fig, G_fig)
    rng_reg = MersenneTwister(99)
    tf_indices = sort(randperm(rng_reg, G_fig)[1:4])
    for tf in tf_indices
        n_targets = rand(rng_reg, 3:5)
        possible = setdiff(1:G_fig, [tf])
        targets = possible[randperm(rng_reg, length(possible))[1:n_targets]]
        for tgt in targets
            sign = rand(rng_reg, [-1, 1])
            W_fig[tf, tgt] = sign * (2.0 + 4.0 * rand(rng_reg))
        end
    end

    net_fig = GeneNetwork(basals_cal, W_fig; k_on=k_on_cal, k_off=k_off_cal)
    kin_fig = KineticParams(k_t=2.0, K_d=50.0, n=4.0, mu_m=mu_m_cal, mu_p=0.2, dilution=dilution_cal)
    alg_fig = BinomialTauLeap(0.1)

    println("  Simulating $n_mono cells x $G_fig genes with BinomialTauLeap..."); flush(stdout)
    rng = MersenneTwister(42)
    Y_synth = simulate(net_fig, alg_fig, kin_fig;
                       cell_num=n_mono, T=500.0, readout=:mrna, rng=rng)
    cap_fig = CaptureModel(efficiency=eff, efficiency_std=0.5, readout=:mrna)
    Y_cap_synth = Float64.(apply_capture(Y_synth, cap_fig; rng=MersenneTwister(123)))

    # ── Per-gene stats ──
    real_means = vec(mean(X_sel, dims=1))
    real_vars  = vec(var(X_sel, dims=1))
    real_fanos = real_vars ./ max.(real_means, 1e-6)
    synth_means = vec(mean(Y_cap_synth, dims=1))
    synth_vars  = vec(var(Y_cap_synth, dims=1))
    synth_fanos = synth_vars ./ max.(synth_means, 1e-6)

    println("  Diagnostics:"); flush(stdout)
    println("    Real  — sparsity=$(round(sum(X_sel .== 0)/length(X_sel), digits=3)), " *
            "median Fano=$(round(median(real_fanos), digits=2)), " *
            "lib-size CV=$(round(std(vec(sum(X_sel, dims=2)))/max(mean(vec(sum(X_sel, dims=2))),1e-6), digits=3))"); flush(stdout)
    println("    Synth — sparsity=$(round(sum(Y_cap_synth .== 0)/length(Y_cap_synth), digits=3)), " *
            "median Fano=$(round(median(synth_fanos), digits=2)), " *
            "lib-size CV=$(round(std(vec(sum(Y_cap_synth, dims=2)))/max(mean(vec(sum(Y_cap_synth, dims=2))),1e-6), digits=3))"); flush(stdout)

    # ── Figure 5: 3x3 layout ──
    # Top two rows: 6 gene density overlays (high/medium/low variance)
    # Bottom row: mean-variance scatter + Fano scatter

    # Pick 6 representative genes: rank 1,5 (high), 20,25 (mid), 40,45 (low)
    repr_ranks = [1, 5, 20, 25, 40, 45]
    fig5 = Figure(size=(1000, 900))

    for (pi, ri) in enumerate(repr_ranks)
        row = (pi - 1) ÷ 3 + 1
        col = (pi - 1) % 3 + 1
        ax = Axis(fig5[row, col], xlabel="Count", ylabel="Density",
                  title=gene_labels[ri])

        real_counts = X_sel[:, ri]
        synth_counts = Y_cap_synth[:, ri]

        maxval = max(maximum(real_counts), maximum(synth_counts))
        bins = 0:1:max(maxval, 1)

        hist!(ax, real_counts, bins=bins, normalization=:pdf,
              color=(:steelblue, 0.5), label="Real")
        hist!(ax, synth_counts, bins=bins, normalization=:pdf,
              color=(:firebrick, 0.5), label="Synthetic")

        if pi == 1
            axislegend(ax, position=:rt, labelsize=10)
        end
    end

    # Bottom-left: mean-variance scatter (log-log)
    ax_mv = Axis(fig5[3, 1:2], xlabel="Mean", ylabel="Variance",
                 title="Mean–variance (per gene)", xscale=log10, yscale=log10)
    scatter!(ax_mv, real_means, real_vars, color=(:steelblue, 0.7),
             markersize=8, label="Real")
    scatter!(ax_mv, synth_means, synth_vars, color=(:firebrick, 0.7),
             markersize=8, label="Synthetic")
    # Poisson reference
    mv_range = [minimum(filter(>(0), [real_means; synth_means])),
                maximum([real_means; synth_means])]
    lines!(ax_mv, mv_range, mv_range, color=:grey50, linestyle=:dash, linewidth=1)
    axislegend(ax_mv, position=:lt, labelsize=10)

    # Bottom-right: Fano factor comparison (real vs synth per gene, log-log)
    ax_fano = Axis(fig5[3, 3], xlabel="Real Fano", ylabel="Synthetic Fano",
                   title="Fano factor (per gene)",
                   xscale=log10, yscale=log10)
    scatter!(ax_fano, real_fanos, synth_fanos, color=:black, markersize=6)
    fano_lo = minimum(filter(>(0), [real_fanos; synth_fanos])) * 0.8
    fano_hi = max(maximum(real_fanos), maximum(synth_fanos)) * 1.2
    lines!(ax_fano, [fano_lo, fano_hi], [fano_lo, fano_hi],
           color=:grey50, linestyle=:dash, linewidth=1)

    save(joinpath(figdir, "real_vs_synthetic.png"), fig5, px_per_unit=3)
    println("  Saved real_vs_synthetic.png"); flush(stdout)
else
    println("  PBMC 3K data not found. Run experiments/compare_real_scrnaseq.jl first."); flush(stdout)
end

# ═══════════════════════════════════════════════════════════════
#  Figure 6: Analytical validation
#  (a) Protein Fano factor vs translation rate — all algorithms
#  (b) Grima LNA breakdown: CLE vs tau-leap vs SSA
#  (c) Thomas volume distribution vs analytical PDF
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 6: analytical validation..."); flush(stdout)

fig6 = Figure(size=(1100, 800))

beta_val = 2.0; mu_m_val = 0.1; mu_p_val = 0.2

# ── Panel (a): Fano vs k_t sweep ──
k_ts = [0.5, 1.0, 2.0, 4.0]

alg_specs = [
    ("SSA",               :black,      :circle),
    ("BinomialTauLeap",   :steelblue,  :utriangle),
    ("CLE",               :firebrick,  :rect),
]

net_val = GeneNetwork(1, [beta_val], zeros(1, 1))
n_cells_val = 10000

ax6a = Axis(fig6[1, 1], xlabel="Translation rate (k_t)",
            ylabel="Protein Fano factor",
            title="(a) Fano factor vs translation rate")

kt_fine = range(0.3, 4.5, length=50)
lines!(ax6a, kt_fine, 1.0 .+ kt_fine ./ (mu_m_val + mu_p_val),
       color=:grey40, linewidth=2, linestyle=:dash, label="Analytical")

for (alg_name, color, marker) in alg_specs
    fanos_sim = Float64[]
    for kt in k_ts
        kin_v = KineticParams(k_t=kt, K_d=50.0, n=2.0, mu_m=mu_m_val, mu_p=mu_p_val)
        alg_v = if alg_name == "SSA"
            SSA()
        elseif alg_name == "BinomialTauLeap"
            BinomialTauLeap(0.05)
        else
            CLE(0.05)
        end
        Y_v = simulate(net_val, alg_v, kin_v;
                       cell_num=n_cells_val, T=500.0, readout=:protein,
                       rng=MersenneTwister(42))
        p_counts = vec(Y_v)
        push!(fanos_sim, var(p_counts) / mean(p_counts))
    end
    scatter!(ax6a, k_ts, fanos_sim, color=color, marker=marker,
             markersize=12, label=alg_name)
end
axislegend(ax6a, position=:lt, labelsize=10)

# ── Panel (b): Grima LNA breakdown ──
ax6b = Axis(fig6[1, 2],
            xlabel="Algorithm",
            ylabel="Protein Fano factor",
            title="(b) LNA breakdown at low counts",
            xticks=(1:4, ["SSA", "PoissonTL", "BinomialTL", "CLE"]))

kin_grima = KineticParams(k_t=1.0, K_d=50.0, n=2.0, mu_m=mu_m_val, mu_p=mu_p_val)
expected_fano = 1.0 + 1.0 / (mu_m_val + mu_p_val)  # 4.333
n_cells_grima = 15000

alg_grima = [("SSA", SSA()),
             ("PoissonTL", PoissonTauLeap(0.05)),
             ("BinomialTL", BinomialTauLeap(0.05)),
             ("CLE", CLE(0.05))]

for (beta_g, label_g, dodge, color_g) in [
        (0.1, "<m> ≈ 1", -0.15, :steelblue),
        (10.0, "<m> ≈ 100", 0.15, :firebrick)]

    net_g = GeneNetwork(1, [beta_g], zeros(1, 1))
    fanos_g = Float64[]

    for (alg_name, alg_g) in alg_grima
        Y_g = simulate(net_g, alg_g, kin_grima;
                       cell_num=n_cells_grima, T=500.0, readout=:protein,
                       rng=MersenneTwister(42))
        p_g = vec(Y_g)
        push!(fanos_g, var(p_g) / max(mean(p_g), 1e-6))
    end
    scatter!(ax6b, (1:4) .+ dodge, fanos_g, color=color_g,
             markersize=12, label=label_g)
end

hlines!(ax6b, [expected_fano], color=:grey40, linestyle=:dash, linewidth=2,
        label="Analytical ($(round(expected_fano, digits=2)))")
axislegend(ax6b, position=:lb, labelsize=9)

# ── Panel (c): mRNA Fano vs mean expression — CLE breakdown ──
# mRNA is Poisson (Fano=1) regardless of mean. CLE deviates at low counts.

println("  Running mRNA Fano sweep..."); flush(stdout)
betas_sweep = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

ax6c = Axis(fig6[2, 1], xlabel="Mean mRNA (<m> = β/μ_m)",
            ylabel="mRNA Fano factor",
            title="(c) mRNA Fano vs mean expression",
            xscale=log10)

for (alg_name, alg_fn, n_cells_s, color, marker) in [
        ("SSA",        () -> SSA(),                 5000,  :black,     :circle),
        ("BinomialTL", () -> BinomialTauLeap(0.05), 10000, :steelblue, :utriangle),
        ("CLE",        () -> CLE(0.05),             10000, :firebrick, :rect)]
    means_s = Float64[]
    fanos_s = Float64[]
    for b in betas_sweep
        net_s = GeneNetwork(1, [b], zeros(1, 1))
        kin_s = KineticParams(k_t=1.0, K_d=50.0, n=2.0, mu_m=mu_m_val, mu_p=mu_p_val)
        Y_s = simulate(net_s, alg_fn(), kin_s;
                       cell_num=n_cells_s, T=500.0, readout=:mrna,
                       rng=MersenneTwister(42))
        m_s = vec(Y_s)
        push!(means_s, mean(m_s))
        push!(fanos_s, var(m_s) / max(mean(m_s), 1e-6))
    end
    scatter!(ax6c, means_s, fanos_s, color=color, marker=marker,
             markersize=10, label=alg_name)
end

hlines!(ax6c, [1.0], color=:grey40, linewidth=2, linestyle=:dash,
        label="Poisson (F=1)")
axislegend(ax6c, position=:rb, labelsize=9)

# ── Panel (d): Thomas — volume distribution vs analytical PDF ──

println("  Running population simulation for volume distribution..."); flush(stdout)
V_div_val = 2.0
net_pop_val = GeneNetwork(1, [beta_val], zeros(1, 1))
kin_pop_val = KineticParams(k_t=1.0, K_d=50.0, n=2.0, mu_m=mu_m_val, mu_p=mu_p_val)
pop_val = PopulationConfig(cell_num=5000, growth_rate=0.03,
                           V_div=V_div_val, V_init=(1.0, 2.0), div_check_interval=1)

_, state_val = simulate_with_state(net_pop_val, BinomialTauLeap(0.1), kin_pop_val, pop_val;
                                   T=500.0, readout=:both, rng=MersenneTwister(42))
V_sim = state_val.volumes

ax6d = Axis(fig6[2, 2], xlabel="Cell volume", ylabel="Count",
            title="(d) Steady-state volume distribution")

hist!(ax6d, V_sim, bins=range(V_div_val/2, V_div_val, length=30),
      color=(:steelblue, 0.4), strokecolor=:steelblue, strokewidth=0.5)

V_mean_sim = mean(V_sim)
V_median_sim = median(V_sim)

vlines!(ax6d, [V_mean_sim], color=:black, linewidth=2, linestyle=:dash,
        label="Mean ($(round(V_mean_sim, digits=2)))")
vlines!(ax6d, [V_median_sim], color=:darkorange, linewidth=2, linestyle=:dashdot,
        label="Median ($(round(V_median_sim, digits=2)))")

xlims!(ax6d, 0.95, 2.05)
axislegend(ax6d, position=:rt, labelsize=9)

save(joinpath(figdir, "analytical_validation.png"), fig6, px_per_unit=3)
println("  Saved analytical_validation.png"); flush(stdout)

println("\nAll figures saved to $figdir"); flush(stdout)
