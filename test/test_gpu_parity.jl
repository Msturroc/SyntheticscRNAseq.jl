using CUDA

@testset "GPU parity" begin
    if !CUDA.functional()
        @info "Skipping GPU tests (CUDA not functional)"
        @test true
        return
    end

    @info "Running GPU tests on $(CUDA.name(CUDA.device()))"

    # ── GPU CLE vs CPU CLE ──
    net = GeneNetwork(1, [5.0], zeros(1, 1))
    kin = KineticParams(k_t=2.0, K_d=50.0, n=2.0, mu_m=0.1, mu_p=0.2)

    rng = MersenneTwister(42)
    Y_cpu = simulate(net, CLE(0.1), kin; cell_num=5000, T=200.0,
                     readout=:protein, rng=rng)

    Y_gpu = simulate(net, CLE(0.1), kin, Val(:gpu);
                     cell_num=5000, T=200.0, readout=:protein)

    # Ensemble stats should match within 5%
    cpu_mean = mean(Y_cpu)
    gpu_mean = mean(Y_gpu)
    @test abs(cpu_mean - gpu_mean) / cpu_mean < 0.05

    cpu_var = var(Y_cpu)
    gpu_var = var(Y_gpu)
    @test abs(cpu_var - gpu_var) / cpu_var < 0.15

    # ── GPU BinomialTauLeap vs CPU ──
    rng = MersenneTwister(100)
    Y_cpu_bin = simulate(net, BinomialTauLeap(0.1), kin;
                         cell_num=5000, T=200.0, readout=:protein, rng=rng)

    Y_gpu_bin = simulate(net, BinomialTauLeap(0.1), kin, Val(:gpu);
                         cell_num=5000, T=200.0, readout=:protein)

    cpu_mean_bin = mean(Y_cpu_bin)
    gpu_mean_bin = mean(Y_gpu_bin)
    @test abs(cpu_mean_bin - gpu_mean_bin) / cpu_mean_bin < 0.05

    # ── GPU batched CLE ──
    nets = [GeneNetwork(1, [Float64(i)], zeros(1, 1)) for i in 2:5]
    results = simulate_gpu_batch(nets, CLE(0.1), kin;
                                 cell_num=1000, T=100.0, readout=:protein)

    @test length(results) == 4
    for k in 1:4
        @test size(results[k]) == (1000, 1)
        @test all(isfinite.(results[k]))
        @test mean(results[k]) > 0
    end
    @test mean(results[4]) > mean(results[1])

    # ── GPU batched BinomialTauLeap ──
    results_bin = simulate_gpu_batch(nets, BinomialTauLeap(0.1), kin;
                                     cell_num=1000, T=100.0, readout=:protein)

    @test length(results_bin) == 4
    for k in 1:4
        @test size(results_bin[k]) == (1000, 1)
        @test all(isfinite.(results_bin[k]))
    end

    # ── Multi-gene GPU ──
    basals = [3.0, 5.0]
    A = [0.0 4.0; -3.0 0.0]
    net2 = GeneNetwork(basals, A)

    Y_gpu2 = simulate(net2, CLE(0.1), kin, Val(:gpu);
                      cell_num=2000, T=100.0, readout=:protein)
    @test size(Y_gpu2) == (2000, 2)
    @test all(isfinite.(Y_gpu2))

    # ── GPU CLE with population ──
    pop = PopulationConfig(cell_num=500, growth_rate=0.03,
                           V_div=2.0, V_init=(0.8, 1.2),
                           div_check_interval=5)

    Y_gpu_pop = simulate(net, CLE(0.1), kin, Val(:gpu), pop;
                         T=100.0, readout=:protein)
    @test size(Y_gpu_pop) == (500, 1)
    @test all(isfinite.(Y_gpu_pop))
    @test mean(Y_gpu_pop) > 0
end
