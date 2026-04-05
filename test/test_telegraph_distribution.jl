#= ================================================================
   Telegraph model: exact distribution validation.

   Tests that SSA and BinomialTauLeap empirical mRNA distributions
   converge to the exact Peccoud-Ycart (1995) steady-state solution
   across four parameter regimes (constitutive, moderate bursting,
   strong bursting, bimodal).

   Uses KS test against exact CDF and moment comparison.
   ================================================================ =#

@testset "Telegraph distribution — exact PMF validation" begin

    MU_M = 0.1
    MU_P = 0.2
    K_T = 1.0
    N_CELLS = 20000
    T_SIM = 1500.0

    function empirical_ks_vs_exact(samples, p_exact)
        cdf_exact = cumsum(p_exact)
        n = length(samples)
        sorted = sort(round.(Int, max.(samples, 0)))
        d_max = 0.0
        for (i, s) in enumerate(sorted)
            ecdf_val = i / n
            idx = min(s + 1, length(cdf_exact))
            exact_cdf = idx > 0 ? cdf_exact[idx] : 0.0
            d_max = max(d_max, abs(ecdf_val - exact_cdf))
            ecdf_before = (i - 1) / n
            d_max = max(d_max, abs(ecdf_before - exact_cdf))
        end
        return d_max
    end

    relerr(obs, exact) = abs(obs - exact) / max(abs(exact), 1e-10)

    # ── Test analytical functions themselves ──
    @testset "Analytical function correctness" begin
        # Known limits: when a,b → ∞ with a/(a+b) fixed, distribution → Poisson
        a, b, c = 500.0, 500.0, 40.0
        @test abs(telegraph_fano(a, b, c) - 1.0) < 0.01  # near-Poisson

        # Distribution sums to 1
        for (a, b, c) in [(2.0, 5.0, 70.0), (0.2, 0.3, 50.0), (0.5, 5.0, 110.0)]
            p = telegraph_distribution(a, b, c)
            @test abs(sum(p) - 1.0) < 1e-6

            # Mean from PMF matches formula
            ns = 0:(length(p)-1)
            @test abs(sum(ns .* p) - telegraph_mean(a, b, c)) < 0.01

            # Variance from PMF matches formula
            μ = sum(ns .* p)
            v = sum((ns .- μ).^2 .* p)
            @test relerr(v, telegraph_variance(a, b, c)) < 0.001
        end
    end

    # ── Test simulator convergence to exact distribution ──
    regimes = [
        (name="Constitutive", k_on=5.0, k_off=5.0, beta=4.0),
        (name="Moderate bursting", k_on=0.2, k_off=0.5, beta=7.0),
        (name="Strong bursting", k_on=0.05, k_off=0.5, beta=11.0),
        (name="Bimodal", k_on=0.02, k_off=0.03, beta=5.0),
    ]

    ks_crit = ks_critical_value(N_CELLS, N_CELLS)

    for regime in regimes
        @testset "$(regime.name)" begin
            a = regime.k_on / MU_M
            b = regime.k_off / MU_M
            c = regime.beta / MU_M

            mean_exact = telegraph_mean(a, b, c)
            fano_exact = telegraph_fano(a, b, c)
            p_exact = telegraph_distribution(a, b, c)

            net = GeneNetwork([regime.beta], zeros(1, 1);
                              k_on=[regime.k_on], k_off=[regime.k_off])
            kin = KineticParams(k_t=K_T, K_d=50.0, n=2.0, mu_m=MU_M, mu_p=MU_P)

            # SSA (exact algorithm)
            rng = MersenneTwister(42)
            Y_ssa = simulate(net, SSA(), kin;
                             cell_num=N_CELLS, T=T_SIM, readout=:mrna, rng=rng)
            samples_ssa = vec(Y_ssa)

            @test relerr(mean(samples_ssa), mean_exact) < 0.05
            fano_ssa = var(samples_ssa) / mean(samples_ssa)
            @test relerr(fano_ssa, fano_exact) < 0.10
            ks_ssa = empirical_ks_vs_exact(samples_ssa, p_exact)
            @test ks_ssa < 3 * ks_crit  # generous margin

            # BinomialTauLeap
            rng = MersenneTwister(42)
            Y_bt = simulate(net, BinomialTauLeap(0.05), kin;
                            cell_num=N_CELLS, T=T_SIM, readout=:mrna, rng=rng)
            samples_bt = vec(Y_bt)

            @test relerr(mean(samples_bt), mean_exact) < 0.05
            fano_bt = var(samples_bt) / mean(samples_bt)
            @test relerr(fano_bt, fano_exact) < 0.15
        end
    end
end
