#= ================================================================
   Holimap validation: compare analytical marginals against
   switching-mode SSA for multi-gene networks (G=2-5).

   Uses weak regulation (small |A|, K_d=100, n=2) to stay in the
   regime where Holimap's 2nd-order moment closure is accurate.

   Generates a multi-panel PMF overlay figure and prints a
   KS/moment comparison table.
   ================================================================ =#

using SyntheticscRNAseq
using Random
using Statistics
using CairoMakie

rng = Random.MersenneTwister(2024)

# Kinetics in the Holimap-accurate regime: gradual Hill (n=2, K_d=100)
kin = KineticParams(k_t=2.0, K_d=100.0, n=2.0,
                    mu_m=0.1, mu_p=0.2, dilution=0.0)

# ── Define test networks ────────────────────────────────────────

function make_network(G::Int, rng::AbstractRNG)
    basals = 2.0 .+ rand(rng, G) .* 2.0  # 2.0–4.0
    A = zeros(G, G)
    k_on = fill(0.2, G)
    k_off = fill(0.3, G)

    # Chain activation: gene j activates gene j+1 (weak)
    for j in 1:G-1
        A[j+1, j] = 0.2 + rand(rng) * 0.3  # 0.2–0.5
    end
    # Last gene weakly represses first (creates loop for G ≥ 3)
    if G >= 3
        A[1, G] = -(0.2 + rand(rng) * 0.3)
    end

    return GeneNetwork(basals, A; k_on=k_on, k_off=k_off)
end

# ── Run validation ──────────────────────────────────────────────

networks = Dict{Int, GeneNetwork}()
holimap_results = Dict{Int, HolimapResult}()
sim_data = Dict{Int, Matrix{Float64}}()

n_cells = 20000
T_sim = 500.0

for G in 2:5
    println("━━━ G = $G ━━━")

    net = make_network(G, copy(rng))
    networks[G] = net

    # Holimap
    print("  Holimap... ")
    t_hm = @elapsed begin
        result = holimap_marginals(net, kin; nmax=200)
    end
    holimap_results[G] = result
    println("$(round(t_hm, digits=3))s, converged=$(result.converged), iters=$(result.iterations)")

    # BinomialTauLeap (switching mode)
    print("  BinomialTauLeap ($n_cells cells, T=$T_sim)... ")
    t_sim = @elapsed begin
        Y = simulate(net, BinomialTauLeap(0.05), kin;
                     cell_num=n_cells, T=T_sim, readout=:mrna,
                     rng=copy(rng), regulation_mode=:switching)
    end
    sim_data[G] = Y
    println("$(round(t_sim, digits=2))s")
end

# ── Print comparison table ──────────────────────────────────────

println("\n┌────────┬──────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
println("│   G    │ Gene │ HM mean  │ Sim mean │ HM var   │ Sim var  │ KS stat  │")
println("├────────┼──────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

for G in 2:5
    result = holimap_results[G]
    Y = sim_data[G]
    for gene in 1:G
        sim_mean = mean(Y[:, gene])
        sim_var = var(Y[:, gene])

        # KS statistic
        samples = sort(Y[:, gene])
        pmf = result.marginals[gene]
        cdf_an = cumsum(pmf)
        ks = 0.0
        for (k, x) in enumerate(samples)
            idx = round(Int, x) + 1
            F_an = idx <= length(cdf_an) ? cdf_an[idx] : 1.0
            F_emp = k / length(samples)
            ks = max(ks, abs(F_an - F_emp))
        end

        println("│ G=$(lpad(G, 2))   │  $(lpad(gene, 2))  │ $(lpad(round(result.means[gene], digits=2), 8)) │ $(lpad(round(sim_mean, digits=2), 8)) │ $(lpad(round(result.variances[gene], digits=2), 8)) │ $(lpad(round(sim_var, digits=2), 8)) │ $(lpad(round(ks, digits=4), 8)) │")
    end
end
ks_crit = 1.36 / sqrt(n_cells)
println("└────────┴──────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
println("KS critical value (α=0.05, n=$n_cells): $(round(ks_crit, digits=4))")

# ── Generate figure ─────────────────────────────────────────────

fig = Figure(size=(1200, 900))

for (row, G) in enumerate(2:5)
    local result = holimap_results[G]
    local Y = sim_data[G]

    for gene in 1:min(G, 3)  # show up to 3 genes per row
        local col = gene
        local ax = Axis(fig[row, col],
                        title="G=$G, gene $gene",
                        xlabel="mRNA count",
                        ylabel="Probability")

        # SSA histogram (normalized)
        local sim_counts = Y[:, gene]
        local max_n = round(Int, maximum(sim_counts))
        local bins = 0:max_n
        local hist_counts = zeros(length(bins))
        for x in sim_counts
            local idx = round(Int, x) + 1
            if 1 <= idx <= length(hist_counts)
                hist_counts[idx] += 1.0
            end
        end
        hist_counts ./= sum(hist_counts)

        barplot!(ax, collect(bins), hist_counts, color=(:steelblue, 0.4),
                 label="SSA")

        # Holimap PMF
        local pmf = result.marginals[gene]
        local n_vals = 0:length(pmf)-1
        lines!(ax, collect(n_vals), pmf, color=:red, linewidth=2,
               label="Holimap")

        xlims!(ax, -0.5, min(max_n + 5, length(pmf)))
        if row == 1 && gene == 1
            axislegend(ax, position=:rt)
        end
    end
end

Label(fig[0, :], "Holimap vs switching-mode SSA: mRNA marginals",
      fontsize=18, font=:bold)

mkpath("figures")
save("figures/holimap_validation.png", fig, px_per_unit=2)
println("\nFigure saved to figures/holimap_validation.png")
