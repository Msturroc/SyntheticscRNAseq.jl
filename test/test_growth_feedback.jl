@testset "Volume-dependent transcription" begin
    # ── Volume scaling: higher V → more transcription ──
    # With volume-dependent transcription, cells with larger volumes
    # should produce more mRNA on average.

    net = GeneNetwork(1, [2.0], zeros(1, 1))
    kin = KineticParams(k_t=1.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.2)

    # Small cells (V ≈ 0.5)
    pop_small = PopulationConfig(cell_num=500, growth_rate=0.001,  # slow growth
                                 V_div=10.0,  # never divides
                                 V_init=(0.4, 0.6),
                                 div_check_interval=100)

    rng = MersenneTwister(42)
    Y_small = simulate(net, BinomialTauLeap(0.1), kin;
                       T=200.0, readout=:mrna, rng=rng, population=pop_small)

    # Large cells (V ≈ 2.0)
    pop_large = PopulationConfig(cell_num=500, growth_rate=0.001,
                                 V_div=10.0,
                                 V_init=(1.8, 2.2),
                                 div_check_interval=100)

    rng = MersenneTwister(42)
    Y_large = simulate(net, BinomialTauLeap(0.1), kin;
                       T=200.0, readout=:mrna, rng=rng, population=pop_large)

    # Larger cells should have more mRNA (volume-dependent transcription)
    @test mean(Y_large) > mean(Y_small) * 1.5

    # ── Without population: no volume effect ──
    rng = MersenneTwister(42)
    Y_nopop = simulate(net, BinomialTauLeap(0.1), kin;
                       cell_num=500, T=200.0, readout=:mrna, rng=rng)

    # Without population, result should be independent of volume
    @test mean(Y_nopop) > 0

    # ── CLE with population produces reasonable output ──
    pop = PopulationConfig(cell_num=200, growth_rate=0.03,
                           V_div=2.0, V_init=(0.8, 1.2),
                           div_check_interval=5)

    rng = MersenneTwister(100)
    Y_cle = simulate(net, CLE(0.1), kin;
                     T=200.0, readout=:protein, rng=rng, population=pop)

    @test size(Y_cle) == (200, 1)
    @test all(isfinite.(Y_cle))
    @test mean(Y_cle) > 0

    # ── Two-stage moments without selection (s=0) ──
    # From Sturrock & Sturrock 2026:
    # <m> = β / μ_m  (at V=1, no growth coupling in transcription at s=0)
    # Fano(protein) = 1 + k_t / (μ_m + μ_p)
    #
    # With population dynamics and volume-dependent transcription,
    # the mean mRNA is β * <V> / μ_m where <V> depends on the
    # growth-division steady state. We can't predict <V> exactly
    # without more theory, but the Fano factor relationship should
    # approximately hold since it's a ratio.

    beta = 3.0
    net_simple = GeneNetwork(1, [beta], zeros(1, 1))
    kin_simple = KineticParams(k_t=2.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.2)
    pop_simple = PopulationConfig(cell_num=2000, growth_rate=0.02,
                                  V_div=2.0, V_init=(0.9, 1.1),
                                  div_check_interval=5)

    rng = MersenneTwister(555)
    Y_both = simulate(net_simple, BinomialTauLeap(0.05), kin_simple;
                      T=500.0, readout=:both, rng=rng, population=pop_simple)

    mrna = Y_both[:, 1]
    protein = Y_both[:, 2]

    # mRNA should be positive
    @test mean(mrna) > 5.0

    # Protein should be positive and larger than mRNA (k_t > 1)
    @test mean(protein) > mean(mrna)

    # Protein Fano factor: should be > 1 (super-Poissonian due to
    # two-stage noise + population heterogeneity)
    fano_p = var(protein) / mean(protein)
    @test fano_p > 1.0

    # mRNA Fano factor: approximately Poisson-like within each cell,
    # but population volume heterogeneity inflates it
    fano_m = var(mrna) / mean(mrna)
    @test fano_m > 0.5  # Should be at least sub-Poisson territory
end
