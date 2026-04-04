@testset "Tau-leap convergence to SSA" begin
    # Single gene, no regulation — clean convergence test
    beta = 3.0
    net = GeneNetwork(1, [beta], zeros(1, 1))
    kin = KineticParams(k_t=2.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.2)

    n_cells = 8000
    T = 300.0

    # SSA reference (ground truth)
    rng = MersenneTwister(42)
    Y_ssa = simulate(net, SSA(), kin; cell_num=n_cells, T=T, readout=:mrna, rng=rng)
    ssa_mean = mean(Y_ssa)

    # Poisson tau-leap at decreasing dt
    dts = [1.0, 0.5, 0.1, 0.05]
    poisson_errors = Float64[]
    binomial_errors = Float64[]
    midpoint_errors = Float64[]

    for dt in dts
        rng = MersenneTwister(100)
        Y_poisson = simulate(net, PoissonTauLeap(dt), kin;
                             cell_num=n_cells, T=T, readout=:mrna, rng=rng)

        rng = MersenneTwister(100)
        Y_binomial = simulate(net, BinomialTauLeap(dt), kin;
                              cell_num=n_cells, T=T, readout=:mrna, rng=rng)

        rng = MersenneTwister(100)
        Y_midpoint = simulate(net, MidpointTauLeap(dt), kin;
                              cell_num=n_cells, T=T, readout=:mrna, rng=rng)

        push!(poisson_errors, abs(mean(Y_poisson) - ssa_mean) / ssa_mean)
        push!(binomial_errors, abs(mean(Y_binomial) - ssa_mean) / ssa_mean)
        push!(midpoint_errors, abs(mean(Y_midpoint) - ssa_mean) / ssa_mean)
    end

    # At finest dt, all methods should be close to SSA
    @test poisson_errors[end] < 0.05
    @test binomial_errors[end] < 0.05
    @test midpoint_errors[end] < 0.05

    # Errors should generally decrease as dt decreases
    @test poisson_errors[end] < poisson_errors[1]
    @test binomial_errors[end] < binomial_errors[1]

    # For high-count single-gene, Poisson and binomial are similar.
    # Binomial's advantage shows mainly at low counts where clamping matters.
    # Just check both converge — no ordering guarantee at high counts.
    @test binomial_errors[end] < 0.10

    # ── 2-gene network convergence ──
    basals = [1.0, 1.0]
    A = [0.0 5.0; -5.0 0.0]
    net2 = GeneNetwork(basals, A)

    rng = MersenneTwister(200)
    Y_ssa2 = simulate(net2, SSA(), kin; cell_num=3000, T=200.0,
                      readout=:protein, rng=rng)

    rng = MersenneTwister(300)
    Y_bin2 = simulate(net2, BinomialTauLeap(0.05), kin;
                      cell_num=3000, T=200.0, readout=:protein, rng=rng)

    # Means should be within 15% for a regulated network
    for g in 1:2
        ssa_m = mean(Y_ssa2[:, g])
        bin_m = mean(Y_bin2[:, g])
        if ssa_m > 1.0  # Only check if mean is non-trivial
            @test abs(ssa_m - bin_m) / ssa_m < 0.15
        end
    end
end
