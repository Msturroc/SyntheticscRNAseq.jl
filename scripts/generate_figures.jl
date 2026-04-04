#= ================================================================
   Generate figures for the README.

   1. Time series: promoter state, mRNA, protein, volume for a
      single cell with telegraph (bursty) transcription
   2. Steady-state distributions: mRNA histogram vs Poisson/NB
   3. Population snapshot: mRNA counts coloured by cell volume
   4. UMAP: synthetic scRNA-seq vs real PBMC 3K monocytes

   Usage:
     julia --project=. scripts/generate_figures.jl
   ================================================================ =#

using Pkg
Pkg.activate(dirname(@__DIR__))

using SyntheticscRNAseq
using Random
using Statistics
using CairoMakie
using UMAP
using SparseArrays

mkpath(joinpath(dirname(@__DIR__), "figures"))
figdir = joinpath(dirname(@__DIR__), "figures")

# ═══════════════════════════════════════════════════════════════
#  Figure 1: Single-cell time series with bursty transcription
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 1: single-cell time series...")

# We need a modified SSA that records the full trajectory.
# Run a single-cell SSA manually and record state at each event.

function ssa_trajectory(net, kin; T=100.0, rng=Random.default_rng())
    G = net.G
    @assert G == 1 "Trajectory recording only for G=1"

    bursty = is_bursty(net)
    m = round(Int, net.basals[1] / (kin.mu_m + kin.dilution))
    p = round(Int, kin.k_t * m / (kin.mu_p + kin.dilution))
    if bursty
        promoter = rand(rng) < net.k_on[1] / (net.k_on[1] + net.k_off[1])
    else
        promoter = true
    end

    ts = Float64[0.0]
    ms = Int[m]
    ps = Int[p]
    prom = Bool[promoter]

    t = 0.0
    while t < T
        reg = 0.0  # G=1, no regulation
        prop_tx = promoter ? max(reg + net.basals[1], 0.0) : 0.0
        prop_mdecay = (kin.mu_m + kin.dilution) * m
        prop_tl = kin.k_t * m
        prop_pdecay = (kin.mu_p + kin.dilution) * p

        a_total = prop_tx + prop_mdecay + prop_tl + prop_pdecay
        if bursty
            prop_on = !promoter ? net.k_on[1] : 0.0
            prop_off = promoter ? net.k_off[1] : 0.0
            a_total += prop_on + prop_off
        end

        a_total <= 0 && break
        tau = -log(rand(rng)) / a_total
        t += tau
        t > T && break

        r = rand(rng) * a_total
        cumsum = 0.0

        # Transcription
        cumsum += prop_tx
        if r < cumsum; m += 1
        else
            cumsum += prop_mdecay
            if r < cumsum; m = max(m-1, 0)
            else
                cumsum += prop_tl
                if r < cumsum; p += 1
                else
                    cumsum += prop_pdecay
                    if r < cumsum; p = max(p-1, 0)
                    elseif bursty
                        cumsum += prop_on
                        if r < cumsum
                            promoter = true
                        else
                            promoter = false
                        end
                    end
                end
            end
        end

        push!(ts, t); push!(ms, m); push!(ps, p); push!(prom, promoter)
    end

    return ts, ms, ps, prom
end

# Bursty gene: visible ON/OFF switching
beta = 5.0; k_on = 0.03; k_off = 0.3
net_ts = GeneNetwork([beta], zeros(1,1); k_on=[k_on], k_off=[k_off])
kin_ts = KineticParams(k_t=2.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.3, dilution=0.02)

rng = MersenneTwister(17)  # pick a seed with nice visible bursts
ts, ms, ps, prom = ssa_trajectory(net_ts, kin_ts; T=300.0, rng=rng)

fig1 = Figure(size=(900, 600))

ax1 = Axis(fig1[1, 1], ylabel="Promoter", yticks=([0, 1], ["OFF", "ON"]))
# Step function for promoter state
for i in 1:length(ts)-1
    lines!(ax1, [ts[i], ts[i+1]], [Int(prom[i]), Int(prom[i])],
           color=prom[i] ? :seagreen : :grey70, linewidth=2)
end
ylims!(ax1, -0.1, 1.1)
hidexdecorations!(ax1, grid=false)

ax2 = Axis(fig1[2, 1], ylabel="mRNA")
stairs!(ax2, ts, Float64.(ms), color=:steelblue, linewidth=1)
hidexdecorations!(ax2, grid=false)

ax3 = Axis(fig1[3, 1], ylabel="Protein", xlabel="Time")
stairs!(ax3, ts, Float64.(ps), color=:firebrick, linewidth=1)

linkxaxes!(ax1, ax2, ax3)

save(joinpath(figdir, "timeseries_bursty.png"), fig1, px_per_unit=3)
println("  Saved timeseries_bursty.png")

# ═══════════════════════════════════════════════════════════════
#  Figure 2: mRNA distribution — constitutive vs bursty
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 2: mRNA distributions...")

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
println("  Saved mrna_distribution.png")

# ═══════════════════════════════════════════════════════════════
#  Figure 3: Population dynamics — volume and expression
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 3: population snapshot...")

net_pop = GeneNetwork(2, [3.0, 1.5], [0.0 4.0; -3.0 0.0])
kin_pop = KineticParams(k_t=1.0, K_d=50.0, n=4.0, mu_m=0.1, mu_p=0.2)
pop = PopulationConfig(cell_num=2000, growth_rate=0.03,
                       V_div=2.0, V_init=(0.8, 1.2), div_check_interval=5)

rng = MersenneTwister(42)
Y_pop, state = simulate_with_state(net_pop, BinomialTauLeap(0.1), kin_pop, pop;
                                   T=300.0, readout=:both, rng=rng)
V = state.volumes

fig3 = Figure(size=(800, 400))

ax3a = Axis(fig3[1, 1], xlabel="Gene 1 mRNA", ylabel="Gene 2 mRNA",
            title="Population snapshot (colour = cell volume)")
sc = scatter!(ax3a, Y_pop[:, 1], Y_pop[:, 2], color=V,
              colormap=:viridis, markersize=4, alpha=0.6)
Colorbar(fig3[1, 2], sc, label="Volume")

ax3b = Axis(fig3[1, 3], xlabel="Cell volume", ylabel="Count",
            title="Volume distribution")
hist!(ax3b, V, bins=30, color=(:teal, 0.7))
vlines!(ax3b, [pop.V_div / 2, pop.V_div], color=:red, linestyle=:dash, linewidth=1.5)

save(joinpath(figdir, "population_snapshot.png"), fig3, px_per_unit=3)
println("  Saved population_snapshot.png")

# ═══════════════════════════════════════════════════════════════
#  Figure 4: Capture model — before and after
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 4: capture model effect...")

net_cap = GeneNetwork(5, [3.0, 1.5, 5.0, 2.0, 0.8], zeros(5, 5))
kin_cap = KineticParams(k_t=2.0, K_d=50.0, n=4.0, mu_m=0.1, mu_p=0.2, dilution=0.03)

rng = MersenneTwister(42)
Y_true = simulate(net_cap, BinomialTauLeap(0.1), kin_cap;
                  cell_num=2000, T=300.0, readout=:mrna, rng=rng)

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
println("  Saved capture_model.png")

# ═══════════════════════════════════════════════════════════════
#  Figure 5: UMAP — synthetic vs real PBMC 3K monocytes
# ═══════════════════════════════════════════════════════════════

println("Generating Figure 5: UMAP comparison...")

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

    # Simulate matching synthetic data
    basals_umap = gm[top50] .* 0.13 ./ 0.07  # calibrate to match means
    net_umap = GeneNetwork(50, basals_umap, zeros(50, 50))
    kin_umap = KineticParams(k_t=2.0, K_d=50.0, n=4.0, mu_m=0.1, mu_p=0.2, dilution=0.03)
    alg_umap = CLE(0.1)  # fast for G=50

    rng = MersenneTwister(42)
    Y_synth = simulate(net_umap, alg_umap, kin_umap;
                       cell_num=n_mono, T=300.0, readout=:mrna, rng=rng)
    cap_umap = CaptureModel(efficiency=0.07, efficiency_std=0.5, readout=:mrna)
    Y_cap_synth = Float64.(apply_capture(Y_synth, cap_umap; rng=MersenneTwister(123)))

    # Log-normalise (standard scRNA-seq preprocessing)
    function log_normalise(Y)
        lib = vec(sum(Y, dims=2))
        lib[lib .== 0] .= 1
        Y_norm = Y ./ lib .* median(lib)
        return log1p.(Y_norm)
    end

    X_norm = log_normalise(X_sel)
    Y_norm = log_normalise(Y_cap_synth)

    # Combined UMAP
    combined = vcat(X_norm, Y_norm)  # (2*n_mono, 50)
    labels = vcat(fill(1, n_mono), fill(2, n_mono))

    println("  Running UMAP on $(size(combined, 1)) cells x $(size(combined, 2)) genes...")
    res = UMAP.fit(combined', 2; n_neighbors=30, min_dist=0.3)
    embedding = hcat(res.embedding...)  # (2, n_cells)

    fig5 = Figure(size=(700, 500))
    ax5 = Axis(fig5[1, 1], xlabel="UMAP 1", ylabel="UMAP 2",
               title="Real monocytes vs synthetic scRNA-seq")

    real_idx = labels .== 1
    synth_idx = labels .== 2

    scatter!(ax5, embedding[1, real_idx], embedding[2, real_idx],
             color=(:steelblue, 0.4), markersize=3, label="Real (CD14+ mono)")
    scatter!(ax5, embedding[1, synth_idx], embedding[2, synth_idx],
             color=(:firebrick, 0.4), markersize=3, label="Synthetic")

    axislegend(ax5, position=:rb)

    save(joinpath(figdir, "umap_comparison.png"), fig5, px_per_unit=3)
    println("  Saved umap_comparison.png")
else
    println("  PBMC 3K data not found. Run experiments/compare_real_scrnaseq.jl first.")
end

println("\nAll figures saved to $figdir")
