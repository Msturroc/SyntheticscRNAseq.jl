@testset "Binomial vs Poisson tau-leap" begin
    # Key property: binomial tau-leap should NEVER produce negative counts
    # Poisson tau-leap CAN produce negatives before clamping

    net = GeneNetwork(1, [0.5], zeros(1, 1))  # Low expression → small counts
    kin = KineticParams(k_t=1.0, K_d=50.0, n=2.0, mu_m=0.5, mu_p=0.5)

    # Large dt to stress-test non-negativity
    rng = MersenneTwister(42)
    Y_bin = simulate(net, BinomialTauLeap(2.0), kin;
                     cell_num=5000, T=100.0, readout=:both, rng=rng)

    @test all(Y_bin .>= 0)
    @test all(isfinite.(Y_bin))

    # Compare means at a moderate dt
    n_cells = 5000
    dt = 0.1

    rng = MersenneTwister(100)
    Y_poisson = simulate(net, PoissonTauLeap(dt), kin;
                         cell_num=n_cells, T=200.0, readout=:mrna, rng=rng)

    rng = MersenneTwister(200)
    Y_binomial = simulate(net, BinomialTauLeap(dt), kin;
                          cell_num=n_cells, T=200.0, readout=:mrna, rng=rng)

    # Both should give similar means for reasonable dt
    poisson_mean = mean(Y_poisson)
    binomial_mean = mean(Y_binomial)

    if poisson_mean > 0.5
        @test abs(poisson_mean - binomial_mean) / poisson_mean < 0.15
    end

    # ── Higher-count regime: both should agree closely ──
    net_high = GeneNetwork(1, [10.0], zeros(1, 1))
    kin_high = KineticParams(k_t=2.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.2)

    rng = MersenneTwister(300)
    Y_p_high = simulate(net_high, PoissonTauLeap(0.1), kin_high;
                        cell_num=5000, T=200.0, readout=:mrna, rng=rng)

    rng = MersenneTwister(400)
    Y_b_high = simulate(net_high, BinomialTauLeap(0.1), kin_high;
                        cell_num=5000, T=200.0, readout=:mrna, rng=rng)

    @test abs(mean(Y_p_high) - mean(Y_b_high)) / mean(Y_b_high) < 0.05
end
