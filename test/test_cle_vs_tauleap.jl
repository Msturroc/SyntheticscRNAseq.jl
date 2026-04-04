@testset "CLE vs tau-leap cross-validation" begin
    # CLE is a continuous approximation valid for high molecule counts.
    # For high-expression networks (β > 5), CLE and tau-leap should
    # give similar means and covariances.

    # High-expression single gene
    net = GeneNetwork(1, [10.0], zeros(1, 1))
    kin = KineticParams(k_t=2.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.2)

    n_cells = 8000
    T = 300.0

    rng = MersenneTwister(42)
    Y_cle = simulate(net, CLE(0.1), kin;
                     cell_num=n_cells, T=T, readout=:mrna, rng=rng)

    rng = MersenneTwister(100)
    Y_bin = simulate(net, BinomialTauLeap(0.1), kin;
                     cell_num=n_cells, T=T, readout=:mrna, rng=rng)

    # Means should be within 5%
    cle_mean = mean(Y_cle)
    bin_mean = mean(Y_bin)
    @test abs(cle_mean - bin_mean) / bin_mean < 0.05

    # Variances within 15% (CLE noise model is approximate)
    cle_var = var(Y_cle)
    bin_var = var(Y_bin)
    @test abs(cle_var - bin_var) / max(bin_var, 1.0) < 0.15

    # ── 2-gene network ──
    basals = [5.0, 8.0]
    A = [0.0 3.0; -4.0 0.0]
    net2 = GeneNetwork(basals, A)

    rng = MersenneTwister(200)
    Y_cle2 = simulate(net2, CLE(0.1), kin;
                      cell_num=5000, T=200.0, readout=:protein, rng=rng)

    rng = MersenneTwister(300)
    Y_bin2 = simulate(net2, BinomialTauLeap(0.05), kin;
                      cell_num=5000, T=200.0, readout=:protein, rng=rng)

    # Per-gene mean comparison
    for g in 1:2
        cle_m = mean(Y_cle2[:, g])
        bin_m = mean(Y_bin2[:, g])
        if bin_m > 5.0
            @test abs(cle_m - bin_m) / bin_m < 0.10
        end
    end
end
