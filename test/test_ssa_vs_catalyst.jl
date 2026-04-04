@testset "SSA vs analytical solutions" begin
    # ── Birth-death: Poisson steady state ──
    # X: ∅ →[β] X →[μ] ∅
    # Steady-state distribution: Poisson(β/μ)
    # Mean = β/μ, Variance = β/μ

    for (beta, mu) in [(5.0, 0.5), (20.0, 1.0), (1.0, 0.1)]
        expected_mean = beta / mu
        samples = simulate_birth_death(beta, mu, 500.0;
                                       n_samples=20000,
                                       rng=MersenneTwister(hash((beta, mu))))
        obs_mean = mean(samples)
        obs_var = var(samples)

        @test abs(obs_mean - expected_mean) / expected_mean < 0.02
        @test abs(obs_var - expected_mean) / expected_mean < 0.08
    end

    # ── Two-stage: mRNA → protein, Fano factor > 1 ──
    # mRNA: ∅ →[β] m →[μ_m] ∅   => m ~ Poisson(β/μ_m)
    # Protein: m →[k_t] m+p, p →[μ_p] ∅
    # Protein Fano factor: F = 1 + k_t/(μ_m + μ_p)
    # (Thattai & van Oudenaarden, PNAS 2001)
    #
    # Mean protein = k_t * β / (μ_m * μ_p)
    # Variance = mean * (1 + k_t / (μ_m + μ_p))

    beta = 2.0
    k_t = 2.0
    mu_m = 0.1
    mu_p = 0.2

    net = GeneNetwork(1, [beta], zeros(1, 1))
    kin = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)

    rng = MersenneTwister(12345)
    Y = simulate(net, SSA(), kin; cell_num=20000, T=500.0, readout=:both, rng=rng)

    mrna = Y[:, 1]
    protein = Y[:, 2]

    # mRNA steady state
    expected_mrna_mean = beta / mu_m  # = 20
    @test abs(mean(mrna) - expected_mrna_mean) / expected_mrna_mean < 0.03

    # Protein steady state
    expected_prot_mean = k_t * beta / (mu_m * mu_p)  # = 200
    @test abs(mean(protein) - expected_prot_mean) / expected_prot_mean < 0.03

    # Protein Fano factor (noise propagation from mRNA bursting)
    expected_fano = 1.0 + k_t / (mu_m + mu_p)  # = 1 + 2/0.3 ≈ 7.67
    obs_fano = var(protein) / mean(protein)
    @test abs(obs_fano - expected_fano) / expected_fano < 0.10

    # mRNA should be approximately Poisson (Fano ≈ 1)
    mrna_fano = var(mrna) / mean(mrna)
    @test abs(mrna_fano - 1.0) < 0.15

    # ── Mora & Walczak regime: regulated gene ──
    # With Hill regulation active, we can't easily get closed-form solutions,
    # but we CAN verify that activation increases and repression decreases
    # expression relative to the unregulated baseline.

    # Activation: gene 1 activated by gene 2 (constitutive)
    basals = [1.0, 5.0]
    A_act = [0.0 6.0; 0.0 0.0]  # gene 2 activates gene 1
    net_act = GeneNetwork(basals, A_act)

    rng = MersenneTwister(999)
    Y_act = simulate(net_act, SSA(), kin; cell_num=5000, T=300.0,
                     readout=:mrna, rng=rng)

    # Gene 1 should be upregulated relative to basal-only (β1/μ_m = 10)
    @test mean(Y_act[:, 1]) > 10.0  # activated above baseline

    # Repression: gene 1 repressed by gene 2
    A_rep = [0.0 -6.0; 0.0 0.0]
    net_rep = GeneNetwork(basals, A_rep)

    rng = MersenneTwister(888)
    Y_rep = simulate(net_rep, SSA(), kin; cell_num=5000, T=300.0,
                     readout=:mrna, rng=rng)

    # Gene 1 should be downregulated (repression pulls toward zero)
    @test mean(Y_rep[:, 1]) < mean(Y_act[:, 1])
end
