@testset "scRNA-seq capture model" begin
    # Generate some fake counts
    rng = MersenneTwister(42)
    X = round.(Int, rand(rng, 1000, 5) .* 100)  # 1000 cells, 5 genes, counts 0-100

    capture = CaptureModel(efficiency=0.1, efficiency_std=0.2, readout=:mrna)

    rng = MersenneTwister(100)
    Y = apply_capture(Float64.(X), capture; rng=rng)

    # Captured counts should be <= original counts
    @test all(Y .<= X)
    @test all(Y .>= 0)

    # Mean capture rate should be approximately efficiency
    capture_rates = Y ./ max.(X, 1)
    overall_rate = mean(capture_rates[X .> 10])  # Only where counts are meaningful
    @test abs(overall_rate - 0.1) < 0.05

    # With efficiency=1.0, should recover most counts
    capture_perfect = CaptureModel(efficiency=1.0, efficiency_std=0.01, readout=:mrna)
    rng = MersenneTwister(200)
    Y_perfect = apply_capture(Float64.(X), capture_perfect; rng=rng)
    @test mean(Y_perfect ./ max.(X, 1)) > 0.95

    # With efficiency=0.0, should get all zeros
    capture_zero = CaptureModel(efficiency=1e-10, efficiency_std=0.01, readout=:mrna)
    rng = MersenneTwister(300)
    Y_zero = apply_capture(Float64.(X), capture_zero; rng=rng)
    @test sum(Y_zero) < sum(X) * 0.01
end
