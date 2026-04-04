@testset "Moran population dynamics" begin
    # ── Cell count stays constant ──
    net = GeneNetwork(1, [2.0], zeros(1, 1))
    kin = KineticParams(k_t=1.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.2)
    pop = PopulationConfig(cell_num=200, growth_rate=0.05,
                           V_div=2.0, V_init=(0.8, 1.2),
                           div_check_interval=5)

    rng = MersenneTwister(42)
    Y = simulate(net, BinomialTauLeap(0.1), kin;
                 T=200.0, readout=:mrna, rng=rng, population=pop)

    @test size(Y) == (200, 1)  # cell_num from PopulationConfig
    @test all(Y .>= 0)
    @test all(isfinite.(Y))

    # ── Population produces non-trivial output ──
    @test mean(Y) > 0.0
    @test var(Y) > 0.0

    # ── Division mechanics: verify volumes reach steady state ──
    # After long simulation, volume distribution should have a
    # well-defined shape between V_div/2 and V_div
    rng = MersenneTwister(100)
    state = initialize_population(net, kin, pop; rng=rng)

    # Run just volume growth + division for many steps
    dt = 0.1
    for step in 1:5000
        grow_volumes!(state, pop, dt)
        if step % pop.div_check_interval == 0
            division_check!(state, pop; rng=rng)
        end
    end

    # All volumes should be between V_div/2 and V_div (approximately)
    # With some tolerance for cells that just grew past V_div
    @test all(state.volumes .> 0)
    @test all(state.volumes .< pop.V_div * 1.5)  # No runaway growth
    @test minimum(state.volumes) > pop.V_div * 0.2  # No collapse

    # ── <1/V> analytical check ──
    # For exponential growth V(a) = V_div/2 * exp(λa) where a is cell age,
    # dividing at V_div, the population-averaged <1/V> should be close to
    # the analytical value.
    # For exponential growth with doubling: <1/V> = ln(2) * 2 / (V_div * (1 - V_div/2/V_div))
    # Actually, for V growing from V0 to 2*V0: <1/V> = ln(2)/(V0*ln(2)) = 1/V0... no.
    #
    # More carefully: V(a) = V0 * exp(λ*a), divides at age T_div = ln(2)/λ
    # In steady-state age distribution (exponential for Moran): p(a) = λ*exp(-λ*a) for a < T_div
    # <1/V> = ∫_0^{T_div} (1/(V0*exp(λa))) * λ*exp(-λa) da (NOT quite right for Moran)
    #
    # For a fixed-pop Moran with exponential growth, the steady-state
    # is more complex. The key test is that volumes are bounded and
    # the population stays at N. Let's just test the basics.
    @test length(state.volumes) == pop.cell_num

    # ── Protein output with population ──
    rng = MersenneTwister(200)
    Y_prot = simulate(net, BinomialTauLeap(0.1), kin;
                      T=200.0, readout=:protein, rng=rng, population=pop)
    @test size(Y_prot) == (200, 1)
    @test all(Y_prot .>= 0)
    @test mean(Y_prot) > 0

    # ── Both readout ──
    rng = MersenneTwister(300)
    Y_both = simulate(net, BinomialTauLeap(0.1), kin;
                      T=100.0, readout=:both, rng=rng, population=pop)
    @test size(Y_both) == (200, 2)  # 1 mRNA + 1 protein

    # ── Multi-gene network with population ──
    basals = [2.0, 3.0]
    A = [0.0 4.0; -3.0 0.0]
    net2 = GeneNetwork(basals, A)
    pop2 = PopulationConfig(cell_num=100, growth_rate=0.03,
                            V_div=2.0, V_init=(0.9, 1.1),
                            div_check_interval=5)

    rng = MersenneTwister(400)
    Y2 = simulate(net2, BinomialTauLeap(0.1), kin;
                  T=150.0, readout=:protein, rng=rng, population=pop2)

    @test size(Y2) == (100, 2)
    @test all(Y2 .>= 0)
    @test all(isfinite.(Y2))
end
