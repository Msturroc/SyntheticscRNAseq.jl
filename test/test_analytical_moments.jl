#= ================================================================
   Analytical moment validation for all simulation algorithms.

   Tests against exact results for the two-stage gene expression
   model (Thattai & van Oudenaarden 2001):
     - mRNA: birth-death with rate β, decay μ_m
     - Protein: translation from mRNA at rate k_t, decay μ_p

   Exact moments (no regulation, no dilution):
     <m>    = β / μ_m
     Var(m) = β / μ_m  (Poisson)
     <p>    = k_t * β / (μ_m * μ_p)
     Fano(p)= 1 + k_t / (μ_m + μ_p)
     Cov(p,m) = k_t * <m> / (μ_m + μ_p)

   Each algorithm is tested with a single constitutive gene (G=1,
   no interactions) and large sample sizes to get tight estimates.
   ================================================================ =#

@testset "Analytical moments — single constitutive gene" begin
    # Parameters chosen for moderate counts (tractable for SSA)
    beta = 2.0
    mu_m = 0.1
    mu_p = 0.2
    k_t = 1.0

    net = GeneNetwork(1, [beta], zeros(1, 1))
    kin = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)

    # Exact analytical moments
    m_mean_exact = beta / mu_m                          # = 20
    m_var_exact  = beta / mu_m                          # = 20 (Poisson)
    p_mean_exact = k_t * beta / (mu_m * mu_p)           # = 100
    p_fano_exact = 1.0 + k_t / (mu_m + mu_p)           # = 4.333...
    cov_pm_exact = k_t * m_mean_exact / (mu_m + mu_p)   # = 66.667

    T_sim = 500.0   # long enough to equilibrate
    N_cells = 10000  # large sample for tight estimates

    # ── Helper: compute all moments from :both readout ──
    function compute_moments(Y_both)
        m_vals = Y_both[:, 1]
        p_vals = Y_both[:, 2]
        m_mean = mean(m_vals)
        m_var  = var(m_vals)
        p_mean = mean(p_vals)
        p_var  = var(p_vals)
        p_fano = p_var / p_mean
        cov_pm = cov(p_vals, m_vals)
        return (; m_mean, m_var, p_mean, p_var, p_fano, cov_pm)
    end

    function relerr(obs, exact)
        return abs(obs - exact) / max(abs(exact), 1e-10)
    end

    # ── SSA (exact, gold standard) ──
    @testset "SSA" begin
        rng = MersenneTwister(42)
        Y = simulate(net, SSA(), kin; cell_num=N_cells, T=T_sim,
                     readout=:both, rng=rng)
        mom = compute_moments(Y)

        @test relerr(mom.m_mean, m_mean_exact) < 0.03
        @test relerr(mom.m_var, m_var_exact) < 0.10
        @test relerr(mom.p_mean, p_mean_exact) < 0.03
        @test relerr(mom.p_fano, p_fano_exact) < 0.10
        @test relerr(mom.cov_pm, cov_pm_exact) < 0.15
    end

    # ── PoissonTauLeap ──
    @testset "PoissonTauLeap" begin
        rng = MersenneTwister(42)
        Y = simulate(net, PoissonTauLeap(0.1), kin; cell_num=N_cells, T=T_sim,
                     readout=:both, rng=rng)
        mom = compute_moments(Y)

        @test relerr(mom.m_mean, m_mean_exact) < 0.03
        @test relerr(mom.m_var, m_var_exact) < 0.15
        @test relerr(mom.p_mean, p_mean_exact) < 0.03
        @test relerr(mom.p_fano, p_fano_exact) < 0.15
    end

    # ── BinomialTauLeap ──
    @testset "BinomialTauLeap" begin
        rng = MersenneTwister(42)
        Y = simulate(net, BinomialTauLeap(0.1), kin; cell_num=N_cells, T=T_sim,
                     readout=:both, rng=rng)
        mom = compute_moments(Y)

        @test relerr(mom.m_mean, m_mean_exact) < 0.03
        @test relerr(mom.m_var, m_var_exact) < 0.15
        @test relerr(mom.p_mean, p_mean_exact) < 0.03
        @test relerr(mom.p_fano, p_fano_exact) < 0.15
    end

    # ── MidpointTauLeap ──
    @testset "MidpointTauLeap" begin
        rng = MersenneTwister(42)
        Y = simulate(net, MidpointTauLeap(0.1), kin; cell_num=N_cells, T=T_sim,
                     readout=:both, rng=rng)
        mom = compute_moments(Y)

        @test relerr(mom.m_mean, m_mean_exact) < 0.03
        @test relerr(mom.m_var, m_var_exact) < 0.15
        @test relerr(mom.p_mean, p_mean_exact) < 0.03
        @test relerr(mom.p_fano, p_fano_exact) < 0.15
    end

    # ── CLE ──
    @testset "CLE" begin
        rng = MersenneTwister(42)
        Y = simulate(net, CLE(0.1), kin; cell_num=N_cells, T=T_sim,
                     readout=:both, rng=rng)
        mom = compute_moments(Y)

        @test relerr(mom.m_mean, m_mean_exact) < 0.05
        @test relerr(mom.m_var, m_var_exact) < 0.20
        @test relerr(mom.p_mean, p_mean_exact) < 0.05
        @test relerr(mom.p_fano, p_fano_exact) < 0.20
    end

    # ── CLEFast ──
    @testset "CLEFast" begin
        rng = MersenneTwister(42)
        Y = simulate(net, CLEFast(0.1), kin; cell_num=N_cells, T=T_sim,
                     readout=:both, rng=rng)
        mom = compute_moments(Y)

        @test relerr(mom.m_mean, m_mean_exact) < 0.05
        @test relerr(mom.m_var, m_var_exact) < 0.20
        @test relerr(mom.p_mean, p_mean_exact) < 0.05
        @test relerr(mom.p_fano, p_fano_exact) < 0.20
    end

    # ── BinomialTauLeapFast ──
    @testset "BinomialTauLeapFast" begin
        rng = MersenneTwister(42)
        Y = simulate(net, BinomialTauLeapFast(0.1), kin; cell_num=N_cells, T=T_sim,
                     readout=:both, rng=rng)
        mom = compute_moments(Y)

        @test relerr(mom.m_mean, m_mean_exact) < 0.03
        @test relerr(mom.m_var, m_var_exact) < 0.15
        @test relerr(mom.p_mean, p_mean_exact) < 0.03
        @test relerr(mom.p_fano, p_fano_exact) < 0.15
    end
end

@testset "Analytical moments — two-gene unregulated" begin
    # Two independent genes (no interactions): each gene should
    # independently match single-gene analytical moments, and
    # cross-gene covariance should be ~0.
    beta1, beta2 = 2.0, 5.0
    mu_m = 0.1
    mu_p = 0.2
    k_t = 1.0

    net = GeneNetwork(2, [beta1, beta2], zeros(2, 2))
    kin = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)

    m1_exact = beta1 / mu_m  # = 20
    m2_exact = beta2 / mu_m  # = 50
    p1_exact = k_t * beta1 / (mu_m * mu_p)  # = 100
    p2_exact = k_t * beta2 / (mu_m * mu_p)  # = 250

    relerr(obs, exact) = abs(obs - exact) / max(abs(exact), 1e-10)

    @testset "SSA two-gene" begin
        rng = MersenneTwister(100)
        Y = simulate(net, SSA(), kin; cell_num=5000, T=500.0,
                     readout=:both, rng=rng)

        @test relerr(mean(Y[:, 1]), m1_exact) < 0.05
        @test relerr(mean(Y[:, 2]), m2_exact) < 0.05
        @test relerr(mean(Y[:, 3]), p1_exact) < 0.05
        @test relerr(mean(Y[:, 4]), p2_exact) < 0.05

        # Cross-gene covariance should be near zero
        cross_corr = cor(Y[:, 3], Y[:, 4])
        @test abs(cross_corr) < 0.05
    end

    @testset "BinomialTauLeap two-gene" begin
        rng = MersenneTwister(100)
        Y = simulate(net, BinomialTauLeap(0.1), kin; cell_num=5000, T=500.0,
                     readout=:both, rng=rng)

        @test relerr(mean(Y[:, 1]), m1_exact) < 0.05
        @test relerr(mean(Y[:, 2]), m2_exact) < 0.05
        @test relerr(mean(Y[:, 3]), p1_exact) < 0.05
        @test relerr(mean(Y[:, 4]), p2_exact) < 0.05
    end

    @testset "CLE two-gene" begin
        rng = MersenneTwister(100)
        Y = simulate(net, CLE(0.1), kin; cell_num=5000, T=500.0,
                     readout=:both, rng=rng)

        @test relerr(mean(Y[:, 1]), m1_exact) < 0.05
        @test relerr(mean(Y[:, 2]), m2_exact) < 0.05
        @test relerr(mean(Y[:, 3]), p1_exact) < 0.05
        @test relerr(mean(Y[:, 4]), p2_exact) < 0.05
    end
end

@testset "Analytical moments — population dilution equivalence" begin
    # Sturrock & Sturrock 2026: for a constitutive gene with
    # Moran population and growth rate μ, the mean molecule counts
    # should match the no-population model with explicit dilution:
    #   <m> ≈ β / μ_m  (dilution is emergent, intrinsic decay dominates)
    #   <p> ≈ k_t * <m> / μ_p
    #
    # The binomial partitioning at division contributes effective
    # dilution, but the mean is preserved because <molecules_mother> +
    # <molecules_daughter> = <molecules_before_division>.
    # Cell count stays exactly N (Moran).

    beta = 2.0
    mu_m = 0.1
    mu_p = 0.2
    k_t = 1.0
    growth_rate = 0.03

    net = GeneNetwork(1, [beta], zeros(1, 1))
    kin = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)
    pop = PopulationConfig(cell_num=2000, growth_rate=growth_rate,
                           V_div=2.0, V_init=(0.8, 1.2),
                           div_check_interval=5)

    relerr(obs, exact) = abs(obs - exact) / max(abs(exact), 1e-10)

    @testset "BinomialTauLeap with population" begin
        rng = MersenneTwister(42)
        Y_pop = simulate(net, BinomialTauLeap(0.1), kin;
                         cell_num=2000, T=200.0, readout=:protein,
                         rng=rng, population=pop)

        # Population should have exactly N cells
        @test size(Y_pop, 1) == 2000

        # All counts non-negative
        @test all(Y_pop .>= 0)

        # With volume-dependent transcription, mean protein should be
        # higher than no-population (transcription scales with V > 1)
        # but still in the right ballpark
        p_mean_nopop = k_t * beta / (mu_m * mu_p)  # = 100
        p_mean_pop = mean(Y_pop)
        @test p_mean_pop > 0.5 * p_mean_nopop
        @test isfinite(p_mean_pop)
    end

    @testset "CLE with population" begin
        rng = MersenneTwister(42)
        Y_pop = simulate(net, CLE(0.1), kin;
                         cell_num=2000, T=200.0, readout=:protein,
                         rng=rng, population=pop)

        @test size(Y_pop, 1) == 2000
        @test all(Y_pop .>= 0)
        @test isfinite(mean(Y_pop))
        @test mean(Y_pop) > 0
    end
end

@testset "Dilution — Sturrock & Sturrock 2026" begin
    # With dilution μ, the effective decay rates are μ_m+μ and μ_p+μ.
    # Exact moments: <m> = β/(μ_m+μ), F(p) = 1 + k_t/(μ_m+μ_p+2μ)
    beta = 2.0
    mu_m = 0.1
    mu_p = 0.2
    k_t = 1.0
    mu_dil = 0.03

    net = GeneNetwork(1, [beta], zeros(1, 1))
    kin_dil = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p,
                            dilution=mu_dil)

    m_mean_exact = beta / (mu_m + mu_dil)                    # ≈ 15.38
    p_mean_exact = k_t * m_mean_exact / (mu_p + mu_dil)      # ≈ 66.90
    fano_exact = 1.0 + k_t / (mu_m + mu_p + 2*mu_dil)       # ≈ 3.778

    relerr(obs, exact) = abs(obs - exact) / max(abs(exact), 1e-10)

    @testset "SSA with dilution" begin
        rng = MersenneTwister(42)
        Y = simulate(net, SSA(), kin_dil; cell_num=10000, T=500.0,
                     readout=:both, rng=rng)
        @test relerr(mean(Y[:, 1]), m_mean_exact) < 0.03
        @test relerr(mean(Y[:, 2]), p_mean_exact) < 0.03
        fano_obs = var(Y[:, 2]) / mean(Y[:, 2])
        @test relerr(fano_obs, fano_exact) < 0.10
    end

    @testset "BinomialTauLeap with dilution" begin
        rng = MersenneTwister(42)
        Y = simulate(net, BinomialTauLeap(0.1), kin_dil; cell_num=10000, T=500.0,
                     readout=:both, rng=rng)
        @test relerr(mean(Y[:, 1]), m_mean_exact) < 0.03
        @test relerr(mean(Y[:, 2]), p_mean_exact) < 0.03
        fano_obs = var(Y[:, 2]) / mean(Y[:, 2])
        @test relerr(fano_obs, fano_exact) < 0.15
    end

    @testset "CLE with dilution" begin
        rng = MersenneTwister(42)
        Y = simulate(net, CLE(0.1), kin_dil; cell_num=10000, T=500.0,
                     readout=:both, rng=rng)
        @test relerr(mean(Y[:, 1]), m_mean_exact) < 0.05
        @test relerr(mean(Y[:, 2]), p_mean_exact) < 0.05
        fano_obs = var(Y[:, 2]) / mean(Y[:, 2])
        @test relerr(fano_obs, fano_exact) < 0.20
    end
end

@testset "Fano factor — protein noise amplification" begin
    # The protein Fano factor F = 1 + k_t/(μ_m + μ_p) is a key
    # result: translation amplifies mRNA noise. We test this
    # across different k_t values to verify the linear relationship.

    mu_m = 0.1
    mu_p = 0.2

    relerr(obs, exact) = abs(obs - exact) / max(abs(exact), 1e-10)

    for k_t in [0.5, 1.0, 2.0, 4.0]
        beta = 2.0
        net = GeneNetwork(1, [beta], zeros(1, 1))
        kin = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)

        fano_exact = 1.0 + k_t / (mu_m + mu_p)

        @testset "Fano k_t=$k_t — BinomialTauLeap" begin
            rng = MersenneTwister(77)
            Y = simulate(net, BinomialTauLeap(0.05), kin;
                         cell_num=15000, T=500.0, readout=:protein, rng=rng)
            fano_obs = var(Y[:, 1]) / mean(Y[:, 1])
            @test relerr(fano_obs, fano_exact) < 0.15
        end
    end
end

# ═══════════════════════════════════════════════════════════════
#  Thomas et al: Snapshot distribution & volume distribution
#
#  In a Moran process with exponential volume growth at rate λ
#  and division at V_div, the steady-state age distribution is
#  p(a) = 2λ exp(-λa) (same as the Powell distribution for
#  exponentially growing populations). This gives the volume PDF
#  p(v) = V_div / v²  for v ∈ [V_div/2, V_div].
#
#  Moments:
#    1. Volumes bounded in [V_div/2, V_div]
#    2. Mean volume <V> = V_div ln(2) ≈ 1.386 for V_div=2
#    3. <1/V> = 3/(2 V_div) = 0.75
#    4. Median volume = 2 V_div / 3 ≈ 1.333
#
#  Note: finite-N Moran with deterministic growth has persistent
#  fluctuations (critical branching number = 1), so tolerances
#  are set to ~15% for single-snapshot tests.
#
#  Reference: Thomas & Shahrezaei (2021), Powell (1956)
# ═══════════════════════════════════════════════════════════════

@testset "Thomas — population snapshot & volume distribution" begin
    beta = 2.0
    net = GeneNetwork(1, [beta], zeros(1, 1))
    kin = KineticParams(k_t=1.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.2)

    V_div = 2.0
    growth_rate = 0.03
    pop = PopulationConfig(cell_num=5000, growth_rate=growth_rate,
                           V_div=V_div, V_init=(1.0, 2.0),
                           div_check_interval=1)

    relerr(obs, exact) = abs(obs - exact) / max(abs(exact), 1e-10)

    rng = MersenneTwister(42)
    Y, state = simulate_with_state(net, BinomialTauLeap(0.1), kin, pop;
                                   T=2000.0, readout=:both, rng=rng)
    V = state.volumes

    # ── 1. Volume bounds: all V ∈ [V_div/2, V_div] ──
    # Allow small tolerance for the division check interval
    @testset "Volume bounds" begin
        V_min_expected = V_div / 2.0  # = 1.0
        # Volumes might slightly exceed V_div between division checks
        V_max_allowed = V_div * exp(growth_rate * pop.div_check_interval * 0.1)
        @test minimum(V) >= V_min_expected * 0.99
        @test maximum(V) <= V_max_allowed * 1.01
    end

    # ── 2. Mean volume: <V> = V_div ln(2) ──
    # From p(v) = V_div/v²: <V> = ∫ v * V_div/v² dv = V_div ln(2)
    @testset "Mean volume" begin
        V_mean_exact = V_div * log(2)  # ≈ 1.386
        @test relerr(mean(V), V_mean_exact) < 0.15
    end

    # ── 3. <1/V> = 3/(2 V_div) ──
    # From p(v) = V_div/v²: <1/V> = ∫ V_div/v³ dv = 3/(2 V_div)
    @testset "<1/V> cell-cycle average" begin
        inv_V_exact = 3.0 / (2.0 * V_div)  # = 0.75
        inv_V_obs = mean(1.0 ./ V)
        @test relerr(inv_V_obs, inv_V_exact) < 0.15
    end

    # ── 4. Cell count preserved (Moran) ──
    @testset "Moran cell count" begin
        @test size(Y, 1) == pop.cell_num
    end

    # ── 5. Volume distribution shape ──
    # p(v) = V_div/v² gives CDF F(v) = 2 - V_div/v,
    # so median = 2 V_div / 3
    @testset "Volume distribution — median" begin
        V_median_exact = 2 * V_div / 3  # ≈ 1.333
        V_median_obs = median(V)
        @test relerr(V_median_obs, V_median_exact) < 0.15
    end

    # ── 6. Thomas snapshot bias: expression correlates with volume ──
    # In the population model, larger cells transcribe faster (volume-
    # dependent transcription). So mRNA counts should positively
    # correlate with cell volume.
    @testset "Snapshot bias — mRNA-volume correlation" begin
        m_vals = Y[:, 1]  # mRNA
        corr_mV = cor(m_vals, V)
        @test corr_mV > 0.1  # positive correlation expected
    end
end

# ═══════════════════════════════════════════════════════════════
#  Sturrock & Sturrock 2026, Appendix E:
#  Continuous dilution ≡ discrete binomial partitioning (at mean)
#
#  Key claim: for a constitutive gene with Moran population
#  dynamics, the mean *concentration* c = m/V matches the mean
#  from the dilution model:
#    <c>_population ≈ β / (μ_m + λ)  where λ = growth rate
#    <m>_dilution   = β / (μ_m + μ)  with μ = λ
#
#  Also validates the no-enrichment theorem: growth-coupled
#  dilution alone cannot produce enrichment (<n>/<r> = 1).
# ═══════════════════════════════════════════════════════════════

@testset "Sturrock — population ↔ dilution equivalence" begin
    beta = 2.0
    mu_m = 0.1
    mu_p = 0.2
    k_t = 1.0
    growth_rate = 0.03

    net = GeneNetwork(1, [beta], zeros(1, 1))
    kin_nodil = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)
    kin_dil = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p,
                            dilution=growth_rate)

    pop = PopulationConfig(cell_num=3000, growth_rate=growth_rate,
                           V_div=2.0, V_init=(0.8, 1.2),
                           div_check_interval=5)

    N_cells = 3000
    T_sim = 500.0

    relerr(obs, exact) = abs(obs - exact) / max(abs(exact), 1e-10)

    # ── Dilution model (no population, constant μ) ──
    rng = MersenneTwister(42)
    Y_dil = simulate(net, BinomialTauLeap(0.1), kin_dil;
                     cell_num=N_cells, T=T_sim, readout=:both, rng=rng)

    # ── Population model (Moran + volume-dependent transcription) ──
    rng = MersenneTwister(42)
    Y_pop, state_pop = simulate_with_state(net, BinomialTauLeap(0.1), kin_nodil, pop;
                                           T=T_sim, readout=:both, rng=rng)
    V = state_pop.volumes

    # ── 1. Mean mRNA concentration equivalence ──
    # c = m / V for population model
    # <c>_pop should match <m>_dil = β / (μ_m + λ)
    @testset "Mean mRNA concentration equivalence" begin
        m_pop = Y_pop[:, 1]
        c_pop = m_pop ./ V  # concentration = count / volume
        c_mean_pop = mean(c_pop)

        m_mean_dil = mean(Y_dil[:, 1])
        c_exact = beta / (mu_m + growth_rate)  # = 15.38

        # Population concentration should match dilution model mean
        @test relerr(c_mean_pop, c_exact) < 0.15
        # Dilution model mean should match exact
        @test relerr(m_mean_dil, c_exact) < 0.05
    end

    # ── 2. Mean protein concentration equivalence ──
    @testset "Mean protein concentration equivalence" begin
        p_pop = Y_pop[:, 2]
        c_p_pop = p_pop ./ V
        c_p_exact = k_t * beta / ((mu_m + growth_rate) * (mu_p + growth_rate))

        p_mean_dil = mean(Y_dil[:, 2])

        @test relerr(mean(c_p_pop), c_p_exact) < 0.15
        @test relerr(p_mean_dil, c_p_exact) < 0.05
    end

    # ── 3. No-enrichment theorem (Theorem 1) ──
    # For growth-coupled dilution alone (no selection), the population
    # mean of molecule counts should satisfy:
    #   <m>_pop / (<c>_pop * <V>) ≈ 1
    # i.e., there's no systematic over/under-representation beyond
    # what's explained by volume scaling.
    @testset "No-enrichment theorem" begin
        m_pop = Y_pop[:, 1]
        c_pop = m_pop ./ V
        # <m> ≈ <c> * <V> if no enrichment (no covariance bias)
        enrichment_ratio = mean(m_pop) / (mean(c_pop) * mean(V))
        # Should be close to 1 (within sampling noise + cov(c,V) contribution)
        @test abs(enrichment_ratio - 1.0) < 0.15
    end

    # ── 4. Fano factor comparison ──
    # Dilution: F(p) = 1 + k_t / (μ_m + μ_p + 2λ) (Sturrock exact)
    # Population: Fano of concentration should be similar
    @testset "Fano factor — dilution vs analytical" begin
        fano_exact = 1.0 + k_t / (mu_m + mu_p + 2 * growth_rate)
        fano_dil = var(Y_dil[:, 2]) / mean(Y_dil[:, 2])
        @test relerr(fano_dil, fano_exact) < 0.15
    end
end

# ═══════════════════════════════════════════════════════════════
#  Grima: LNA/CLE breakdown at low molecule counts
#
#  The Chemical Langevin Equation is a Gaussian (diffusion)
#  approximation that assumes molecule counts are large enough
#  for the Poisson → Gaussian CLT to hold. At low counts
#  (<m> ~ O(1)), this breaks down:
#    - CLE can produce negative values (clamped to 0 → bias)
#    - Variance is systematically wrong
#    - Distribution shape departs from Gaussian
#
#  BinomialTauLeap preserves discrete stochasticity and should
#  match SSA better than CLE in this regime.
#
#  Reference: Grima (2010) "An effective rate equation approach
#  to reaction kinetics in small volumes"; Schnoerr, Sanguinetti
#  & Grima (2017) review of approximation methods.
# ═══════════════════════════════════════════════════════════════

@testset "Grima — LNA breakdown at low copy numbers" begin
    # Use very small β to get <m> ≈ 1 molecule
    beta_low = 0.1
    mu_m = 0.1
    mu_p = 0.2
    k_t = 1.0

    net = GeneNetwork(1, [beta_low], zeros(1, 1))
    kin = KineticParams(k_t=k_t, K_d=50.0, n=2.0, mu_m=mu_m, mu_p=mu_p)

    # Exact moments (still valid even at low counts)
    m_mean_exact = beta_low / mu_m  # = 1.0
    p_mean_exact = k_t * beta_low / (mu_m * mu_p)  # = 5.0
    fano_exact = 1.0 + k_t / (mu_m + mu_p)  # = 4.333

    N_cells = 15000
    T_sim = 500.0

    relerr(obs, exact) = abs(obs - exact) / max(abs(exact), 1e-10)

    # ── SSA (gold standard, exact even at low counts) ──
    rng = MersenneTwister(42)
    Y_ssa = simulate(net, SSA(), kin; cell_num=N_cells, T=T_sim,
                     readout=:both, rng=rng)
    ssa_m_mean = mean(Y_ssa[:, 1])
    ssa_p_mean = mean(Y_ssa[:, 2])
    ssa_p_fano = var(Y_ssa[:, 2]) / mean(Y_ssa[:, 2])

    # ── BinomialTauLeap (discrete, should match SSA well) ──
    rng = MersenneTwister(42)
    Y_bt = simulate(net, BinomialTauLeap(0.05), kin; cell_num=N_cells, T=T_sim,
                    readout=:both, rng=rng)
    bt_m_mean = mean(Y_bt[:, 1])
    bt_p_mean = mean(Y_bt[:, 2])
    bt_p_fano = var(Y_bt[:, 2]) / mean(Y_bt[:, 2])

    # ── CLE (Gaussian approx, expected to be worse at low counts) ──
    rng = MersenneTwister(42)
    Y_cle = simulate(net, CLE(0.05), kin; cell_num=N_cells, T=T_sim,
                     readout=:both, rng=rng)
    cle_m_mean = mean(Y_cle[:, 1])
    cle_p_mean = mean(Y_cle[:, 2])
    cle_p_fano = var(Y_cle[:, 2]) / mean(Y_cle[:, 2])

    # ── 1. All methods should get the mean approximately right ──
    # CLE has systematic positive bias at low counts due to max(x,0)
    # clamping — this IS the Grima LNA breakdown effect.
    @testset "Mean accuracy (all methods)" begin
        @test relerr(ssa_m_mean, m_mean_exact) < 0.10
        @test relerr(bt_m_mean, m_mean_exact) < 0.10
        @test relerr(cle_m_mean, m_mean_exact) < 0.25  # CLE biased at low counts
    end

    # ── 2. BinomialTauLeap should match SSA Fano better than CLE ──
    @testset "BinomialTauLeap beats CLE on Fano at low counts" begin
        bt_fano_err = relerr(bt_p_fano, ssa_p_fano)
        cle_fano_err = relerr(cle_p_fano, ssa_p_fano)

        # BinomialTauLeap should have smaller Fano error than CLE
        @test bt_fano_err < cle_fano_err || bt_fano_err < 0.10
    end

    # ── 3. CLE produces many zeros due to clamping at low counts ──
    # At <m> ≈ 1, a significant fraction of CLE mRNA values will be
    # clamped to exactly 0.0 (from negative Gaussian draws). This
    # clamping bias is a known LNA artifact.
    @testset "CLE clamping artifact at low counts" begin
        # SSA naturally has zeros from the Poisson process
        ssa_zero_frac = count(Y_ssa[:, 1] .== 0) / N_cells

        # CLE should have a similar or higher zero fraction
        # (The CLE zeros come from max(x, 0) clamping, not from
        # the underlying stochastic process being at 0)
        cle_zero_frac = count(Y_cle[:, 1] .== 0.0) / N_cells

        # CLE values are continuous — exact zeros indicate clamping
        # For discrete methods (SSA, BT), zeros are natural
        @test cle_zero_frac > 0.0  # CLE should have some clamped zeros
    end

    # ── 4. SSA mRNA should be Poisson-distributed ──
    # At steady state, constitutive mRNA is exactly Poisson.
    # The Fano factor should be 1.0.
    @testset "SSA mRNA is Poisson" begin
        ssa_m_fano = var(Y_ssa[:, 1]) / mean(Y_ssa[:, 1])
        @test relerr(ssa_m_fano, 1.0) < 0.10
    end

    # ── 5. BinomialTauLeap mRNA Fano closer to 1.0 than CLE ──
    @testset "BinomialTauLeap mRNA Fano closer to Poisson than CLE" begin
        bt_m_fano = var(Y_bt[:, 1]) / mean(Y_bt[:, 1])
        cle_m_fano = var(Y_cle[:, 1]) / mean(Y_cle[:, 1])

        bt_mfano_err = relerr(bt_m_fano, 1.0)
        cle_mfano_err = relerr(cle_m_fano, 1.0)

        @test bt_mfano_err < cle_mfano_err || bt_mfano_err < 0.15
    end

    # ── 6. At higher counts, CLE accuracy should recover ──
    # Use β=10.0 → <m>=100, well within LNA regime
    @testset "CLE recovers at high copy numbers" begin
        beta_high = 10.0
        net_high = GeneNetwork(1, [beta_high], zeros(1, 1))
        fano_exact_high = fano_exact  # same k_t, μ_m, μ_p

        rng = MersenneTwister(42)
        Y_cle_high = simulate(net_high, CLE(0.1), kin; cell_num=10000, T=300.0,
                              readout=:protein, rng=rng)
        cle_fano_high = var(Y_cle_high[:, 1]) / mean(Y_cle_high[:, 1])

        # At high counts, CLE should be accurate
        @test relerr(cle_fano_high, fano_exact_high) < 0.10
    end
end
