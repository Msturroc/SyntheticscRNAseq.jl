@testset "SSA vs birth-death analytical" begin
    # ── Simple birth-death process ──
    # Steady state: mean = β/μ, variance = β/μ (Poisson)
    birth_rate = 5.0
    death_rate = 0.5
    expected_mean = birth_rate / death_rate  # = 10.0
    expected_var = birth_rate / death_rate   # = 10.0

    rng = MersenneTwister(42)
    samples = simulate_birth_death(birth_rate, death_rate, 200.0;
                                   n_samples=20000, rng=rng)

    obs_mean = mean(samples)
    obs_var = var(samples)

    @test abs(obs_mean - expected_mean) / expected_mean < 0.03
    @test abs(obs_var - expected_var) / expected_var < 0.10

    # ── Single-gene SSA (no regulation) ──
    # Gene with β=2.0, μ_m=0.1 → mRNA ss = 20
    # k_t=1.0, μ_p=0.2 → protein ss = k_t * 20 / 0.2 = 100
    beta = 2.0
    net = GeneNetwork(1, [beta], zeros(1, 1))
    kin = KineticParams(k_t=1.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.2)

    rng = MersenneTwister(123)
    Y_mrna = simulate(net, SSA(), kin; cell_num=10000, T=500.0,
                      readout=:mrna, rng=rng)

    rng = MersenneTwister(456)
    Y_prot = simulate(net, SSA(), kin; cell_num=10000, T=500.0,
                      readout=:protein, rng=rng)

    # mRNA: mean ≈ β/μ_m = 20, variance ≈ β/μ_m = 20
    @test abs(mean(Y_mrna) - 20.0) / 20.0 < 0.03
    @test abs(var(Y_mrna) - 20.0) / 20.0 < 0.15

    # Protein: mean ≈ k_t * β / (μ_m * μ_p) = 100
    @test abs(mean(Y_prot) - 100.0) / 100.0 < 0.03

    # ── Two-gene network with regulation ──
    # Just test that it runs without error and produces reasonable output
    basals = [1.0, 1.0]
    A = [0.0 5.0; -5.0 0.0]  # gene 2 activates gene 1, gene 1 represses gene 2
    net2 = GeneNetwork(basals, A)
    kin2 = KineticParams(k_t=2.0, K_d=50.0, n=4.0, mu_m=0.1, mu_p=0.2)

    rng = MersenneTwister(789)
    Y2 = simulate(net2, SSA(), kin2; cell_num=500, T=200.0, readout=:both, rng=rng)

    @test size(Y2) == (500, 4)  # 2 mRNA + 2 protein
    @test all(Y2 .>= 0)
    @test all(isfinite.(Y2))
    @test mean(Y2[:, 1]) > 0  # mRNA gene 1
    @test mean(Y2[:, 3]) > 0  # protein gene 1
end
