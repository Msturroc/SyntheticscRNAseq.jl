@testset "Holimap" begin
    rng = Random.MersenneTwister(42)

    # ── Test 1: single-gene recovery ────────────────────────────
    @testset "single-gene recovery" begin
        # Single gene, no interactions → Holimap must exactly match telegraph
        kin = KineticParams(k_t=2.0, K_d=50.0, n=3.0,
                            mu_m=0.1, mu_p=0.2, dilution=0.0)
        basals = [3.0]
        A = zeros(1, 1)
        k_on = [0.05]
        k_off = [0.15]
        net = GeneNetwork(basals, A; k_on=k_on, k_off=k_off)

        result = holimap_marginals(net, kin; nmax=300)

        a_expect = k_on[1] / kin.mu_m
        b_expect = k_off[1] / kin.mu_m
        c_expect = basals[1] / kin.mu_m

        @test result.converged
        @test result.iterations == 0

        # Effective parameters must match input exactly
        @test result.effective_a[1] ≈ a_expect atol=1e-10
        @test result.effective_b[1] ≈ b_expect atol=1e-10
        @test result.effective_c[1] ≈ c_expect atol=1e-10

        # Marginal must match telegraph_distribution
        p_exact = telegraph_distribution(a_expect, b_expect, c_expect; nmax=300)
        n_compare = min(length(result.marginals[1]), length(p_exact))
        @test sum(abs.(result.marginals[1][1:n_compare] .- p_exact[1:n_compare])) < 1e-10

        # Moments must match
        @test result.means[1] ≈ telegraph_mean(a_expect, b_expect, c_expect) atol=1e-10
        @test result.variances[1] ≈ telegraph_variance(a_expect, b_expect, c_expect) atol=1e-10
    end

    # ── Test 2: two-gene activation (weak regulation) ───────────
    @testset "two-gene activation vs SSA" begin
        # Weak regulation with gradual Hill (n=2, K_d=100) keeps Holimap
        # in the regime where 2nd-order moment closure is accurate.
        kin = KineticParams(k_t=2.0, K_d=100.0, n=2.0,
                            mu_m=0.1, mu_p=0.2, dilution=0.0)
        basals = [3.0, 2.5]
        A = zeros(2, 2)
        A[2, 1] = 0.3  # weak activation of gene 2 by protein 1
        k_on = [0.3, 0.2]
        k_off = [0.5, 0.3]
        net = GeneNetwork(basals, A; k_on=k_on, k_off=k_off)

        result = holimap_marginals(net, kin; nmax=200)
        @test result.converged

        Y_ssa = simulate(net, SSA(), kin;
                         cell_num=20000, T=500.0, readout=:mrna,
                         rng=copy(rng), regulation_mode=:switching)

        for gene in 1:2
            ssa_mean = mean(Y_ssa[:, gene])
            ssa_var = var(Y_ssa[:, gene])

            # Moments should agree within 10% (weak regulation regime)
            @test abs(result.means[gene] - ssa_mean) / max(ssa_mean, 1.0) < 0.10
            @test abs(result.variances[gene] - ssa_var) / max(ssa_var, 1.0) < 0.15

            # KS test on marginal
            ssa_samples = sort(Y_ssa[:, gene])
            pmf = result.marginals[gene]
            cdf_analytical = cumsum(pmf)
            ks_stat = 0.0
            n_samples = length(ssa_samples)
            for (k, x) in enumerate(ssa_samples)
                idx = round(Int, x) + 1
                F_analytical = idx <= length(cdf_analytical) ? cdf_analytical[idx] : 1.0
                F_empirical = k / n_samples
                ks_stat = max(ks_stat, abs(F_analytical - F_empirical))
            end
            ks_crit = 1.36 / sqrt(n_samples)
            @test ks_stat < 8.0 * ks_crit
        end
    end

    # ── Test 3: toggle switch (mutual repression) ───────────────
    @testset "toggle switch vs SSA" begin
        # Weak mutual repression — Holimap ignores cross-gene correlations
        # so tolerance is looser for feedback loops.
        kin = KineticParams(k_t=2.0, K_d=100.0, n=2.0,
                            mu_m=0.1, mu_p=0.2, dilution=0.0)
        basals = [3.0, 3.0]
        A = zeros(2, 2)
        A[1, 2] = -0.2  # weak repression
        A[2, 1] = -0.2
        k_on = [0.3, 0.3]
        k_off = [0.3, 0.3]
        net = GeneNetwork(basals, A; k_on=k_on, k_off=k_off)

        result = holimap_marginals(net, kin; nmax=200)
        @test result.converged

        Y_ssa = simulate(net, SSA(), kin;
                         cell_num=20000, T=500.0, readout=:mrna,
                         rng=copy(rng), regulation_mode=:switching)

        for gene in 1:2
            ssa_mean = mean(Y_ssa[:, gene])
            ssa_var = var(Y_ssa[:, gene])
            @test abs(result.means[gene] - ssa_mean) / max(ssa_mean, 1.0) < 0.15
            @test abs(result.variances[gene] - ssa_var) / max(ssa_var, 1.0) < 0.25
        end
    end

    # ── Test 4: repressilator (3-gene cyclic repression) ────────
    @testset "repressilator vs SSA" begin
        kin = KineticParams(k_t=2.0, K_d=100.0, n=2.0,
                            mu_m=0.1, mu_p=0.2, dilution=0.0)
        basals = [3.0, 3.0, 3.0]
        A = zeros(3, 3)
        A[1, 3] = -0.2
        A[2, 1] = -0.2
        A[3, 2] = -0.2
        k_on = [0.2, 0.2, 0.2]
        k_off = [0.3, 0.3, 0.3]
        net = GeneNetwork(basals, A; k_on=k_on, k_off=k_off)

        result = holimap_marginals(net, kin; nmax=200)
        @test result.converged

        Y_ssa = simulate(net, SSA(), kin;
                         cell_num=20000, T=500.0, readout=:mrna,
                         rng=copy(rng), regulation_mode=:switching)

        for gene in 1:3
            ssa_mean = mean(Y_ssa[:, gene])
            ssa_var = var(Y_ssa[:, gene])
            @test abs(result.means[gene] - ssa_mean) / max(ssa_mean, 1.0) < 0.15
            @test abs(result.variances[gene] - ssa_var) / max(ssa_var, 1.0) < 0.30
        end
    end

    # ── Test 5: fast switching limit → Poisson ──────────────────
    @testset "fast switching limit" begin
        kin = KineticParams(k_t=2.0, K_d=50.0, n=3.0,
                            mu_m=0.1, mu_p=0.2, dilution=0.0)
        basals = [5.0]
        A = zeros(1, 1)
        k_on = [100.0]
        k_off = [100.0]
        net = GeneNetwork(basals, A; k_on=k_on, k_off=k_off)

        result = holimap_marginals(net, kin; nmax=200)
        @test result.converged

        expected_mean = 0.5 * basals[1] / kin.mu_m
        @test abs(result.means[1] - expected_mean) / expected_mean < 0.01

        fano = result.variances[1] / result.means[1]
        @test abs(fano - 1.0) < 0.05
    end

    # ── Test 6: protein moments consistency ─────────────────────
    @testset "protein moments" begin
        kin = KineticParams(k_t=2.0, K_d=50.0, n=3.0,
                            mu_m=0.1, mu_p=0.2, dilution=0.0)
        basals = [3.0]
        A = zeros(1, 1)
        k_on = [0.08]
        k_off = [0.2]
        net = GeneNetwork(basals, A; k_on=k_on, k_off=k_off)

        result = holimap_marginals(net, kin; nmax=300)

        @test result.protein_means[1] ≈ kin.k_t * result.means[1] / kin.mu_p atol=1e-10

        expected_pvar = result.protein_means[1] +
            kin.k_t^2 * result.variances[1] / (kin.mu_p * (kin.mu_m + kin.mu_p))
        @test result.protein_variances[1] ≈ expected_pvar atol=1e-10
    end
end
