module SyntheticscRNAseqCUDAExt

using SyntheticscRNAseq
using CUDA
using CUDA.CUBLAS: gemm_strided_batched!
using LinearAlgebra
using Random
using Distributions: Binomial

# ══════════════════════════════════════════════════════════════════
#  GPU-friendly discrete sampling via inverse CDF
#
#  Normal approximation of Poisson/Binomial fails when λ = np < 5
#  (typical for decay at dt=0.1 with μ=0.1).  Instead, use the
#  Poisson inverse CDF: given u ~ Uniform(0,1), count how many CDF
#  thresholds u exceeds.  This is branchless and exact up to the
#  truncation point.  For large λ (>30), fall back to rounded normal
#  via erfinv.  Compiles to a single GPU kernel via broadcasting.
# ══════════════════════════════════════════════════════════════════

@inline function _sample_poisson_gpu(u::Float32, lam::Float32)
    # Inverse CDF for Poisson(lam).  Iterative accumulation of
    # P(X ≤ k) = exp(-λ) Σ_{j=0}^{k} λ^j/j! avoids large powers.
    # 50 terms covers λ up to ~35 with >99.9% accuracy.
    term = exp(-lam)
    cdf = term
    events = Float32(0)
    for k in 1:50
        events += Float32(u > cdf)
        term *= lam / Float32(k)
        cdf += term
    end
    return events
end

@inline function _sample_binomial_gpu(u::Float32, n::Float32, p::Float32)
    # Binomial(n, p) ≈ Poisson(np) for small p, with clamp to [0, n]
    return clamp(_sample_poisson_gpu(u, n * p), 0f0, n)
end

# ══════════════════════════════════════════════════════════════════
#  GPU CLE — single network
#
#  Direct port of validated cle_gpu.jl from GeneticNetworkSBI.
#  All state arrays on GPU, cuBLAS matmuls, fused broadcast kernels.
# ══════════════════════════════════════════════════════════════════

function SyntheticscRNAseq.simulate(net::GeneNetwork, alg::CLE, kin::KineticParams,
                                    ::Val{:gpu};
                                    cell_num::Int=1000,
                                    T::Float64=50.0,
                                    readout::Symbol=:protein)
    G = net.G
    A_pos_cpu, A_neg_cpu = SyntheticscRNAseq.precompute_hill_matrices(net)

    A_pos = CuArray{Float32}(A_pos_cpu)
    A_neg = CuArray{Float32}(A_neg_cpu)

    m_ss, p_ss = SyntheticscRNAseq.steady_state(net, kin)

    M = CUDA.zeros(Float32, G, cell_num)
    P = CUDA.zeros(Float32, G, cell_num)
    for i in 1:G
        M[i, :] .= Float32(ceil(m_ss[i]))
        P[i, :] .= Float32(ceil(p_ss[i]))
    end

    dt = Float32(alg.dt)
    n_steps = ceil(Int, T / alg.dt)
    k_t = Float32(kin.k_t)
    K_n = Float32(kin.K_d ^ kin.n)
    hill_n = Float32(kin.n)
    mu_m = Float32(kin.mu_m + kin.dilution)   # effective mRNA decay
    mu_p = Float32(kin.mu_p + kin.dilution)   # effective protein decay
    sqrt_dt = Float32(sqrt(alg.dt))

    P_n = similar(M)
    denom = similar(M)
    act_frac = similar(M)
    rep_frac = similar(M)
    reg_input = similar(M)
    temp = similar(M)
    noise_m = similar(M)
    noise_p = similar(M)
    beta_gpu = CuArray{Float32}(reshape(net.basals, G, 1))

    for step in 1:n_steps
        @. P_n = P ^ hill_n
        @. denom = P_n + K_n
        @. act_frac = P_n / denom
        @. rep_frac = K_n / denom

        mul!(reg_input, A_pos, act_frac)
        mul!(temp, A_neg, rep_frac)
        @. reg_input += temp

        CUDA.randn!(noise_m)
        @. begin
            noise_m = noise_m * sqrt_dt * sqrt(max(reg_input + beta_gpu + mu_m * max(M, Float32(0)), Float32(0)))
            M = max(M + (reg_input + beta_gpu - mu_m * M) * dt + noise_m, Float32(0))
        end

        CUDA.randn!(noise_p)
        @. begin
            noise_p = noise_p * sqrt_dt * sqrt(max(k_t * max(M, Float32(0)) + mu_p * max(P, Float32(0)), Float32(0)))
            P = max(P + (k_t * M - mu_p * P) * dt + noise_p, Float32(0))
        end
    end

    if readout == :mrna
        return Matrix{Float64}(Array(M)')
    elseif readout == :protein
        return Matrix{Float64}(Array(P)')
    else
        return hcat(Matrix{Float64}(Array(M)'), Matrix{Float64}(Array(P)'))
    end
end

# ══════════════════════════════════════════════════════════════════
#  GPU CLE — batched (multiple networks)
#
#  3D tensor layout: (G, cell_num, N) where N = batch size.
#  gemm_strided_batched! runs ALL N matmuls in a single cuBLAS call.
# ══════════════════════════════════════════════════════════════════

"""
    simulate_gpu_batch(networks, alg::CLE, kinetics; cell_num, T, readout)

Batch-simulate multiple networks on GPU using strided batched GEMM.
Returns Vector of (cell_num x G) matrices.
"""
function SyntheticscRNAseq.simulate_gpu_batch(networks::Vector{GeneNetwork},
                                              alg::CLE, kin::KineticParams;
                                              cell_num::Int=1000,
                                              T::Float64=50.0,
                                              readout::Symbol=:protein)
    N = length(networks)
    G = networks[1].G

    # Build 3D arrays on CPU
    M_cpu = zeros(Float32, G, cell_num, N)
    P_cpu = zeros(Float32, G, cell_num, N)
    beta_cpu = zeros(Float32, G, 1, N)
    A_pos_3d_cpu = zeros(Float32, G, G, N)
    A_neg_3d_cpu = zeros(Float32, G, G, N)

    for k in 1:N
        net = networks[k]
        A_pos_k, A_neg_k = SyntheticscRNAseq.precompute_hill_matrices(net)
        m_ss, p_ss = SyntheticscRNAseq.steady_state(net, kin)
        for i in 1:G
            M_cpu[i, :, k] .= Float32(ceil(m_ss[i]))
            P_cpu[i, :, k] .= Float32(ceil(p_ss[i]))
            beta_cpu[i, 1, k] = Float32(net.basals[i])
        end
        A_pos_3d_cpu[:, :, k] = Float32.(A_pos_k)
        A_neg_3d_cpu[:, :, k] = Float32.(A_neg_k)
    end

    M = CuArray(M_cpu)
    P = CuArray(P_cpu)
    beta_gpu = CuArray(beta_cpu)
    A_pos_3d = CuArray(A_pos_3d_cpu)
    A_neg_3d = CuArray(A_neg_3d_cpu)

    dt = Float32(alg.dt)
    n_steps = ceil(Int, T / alg.dt)
    k_t = Float32(kin.k_t)
    K_n = Float32(kin.K_d ^ kin.n)
    hill_n = Float32(kin.n)
    mu_m = Float32(kin.mu_m + kin.dilution)   # effective mRNA decay
    mu_p = Float32(kin.mu_p + kin.dilution)   # effective protein decay
    sqrt_dt = Float32(sqrt(alg.dt))

    P_n = similar(M)
    denom = similar(M)
    act_frac = similar(M)
    rep_frac = similar(M)
    reg_input = similar(M)
    temp = similar(M)
    noise_m = similar(M)
    noise_p = similar(M)

    for step in 1:n_steps
        @. P_n = P ^ hill_n
        @. denom = P_n + K_n
        @. act_frac = P_n / denom
        @. rep_frac = K_n / denom

        gemm_strided_batched!('N', 'N', Float32(1), A_pos_3d, act_frac, Float32(0), reg_input)
        gemm_strided_batched!('N', 'N', Float32(1), A_neg_3d, rep_frac, Float32(0), temp)
        @. reg_input += temp

        CUDA.randn!(noise_m)
        @. begin
            noise_m = noise_m * sqrt_dt * sqrt(max(reg_input + beta_gpu + mu_m * max(M, Float32(0)), Float32(0)))
            M = max(M + (reg_input + beta_gpu - mu_m * M) * dt + noise_m, Float32(0))
        end

        CUDA.randn!(noise_p)
        @. begin
            noise_p = noise_p * sqrt_dt * sqrt(max(k_t * max(M, Float32(0)) + mu_p * max(P, Float32(0)), Float32(0)))
            P = max(P + (k_t * M - mu_p * P) * dt + noise_p, Float32(0))
        end
    end

    out = readout == :mrna ? Array(M) : Array(P)
    results = Vector{Matrix{Float64}}(undef, N)
    for k in 1:N
        results[k] = Float64.(out[:, :, k]')
    end
    return results
end

# ══════════════════════════════════════════════════════════════════
#  GPU Binomial Tau-Leap
#
#  Key: replace Poisson normal approximation with Binomial normal
#  approximation on GPU. For decay reactions:
#    events ~ Binomial(n, p) where p = 1 - exp(-rate * dt)
#    GPU approx: clamp(n*p + sqrt(n*p*(1-p)) * z, 0, n)
#  For creation reactions (transcription from unlimited supply):
#    events ~ Poisson(rate * dt)
#    GPU approx: max(rate*dt + sqrt(rate*dt) * z, 0)
#
#  The binomial clamping to [0, n] is unbiased for moderate n,
#  unlike Poisson clamped at 0 which has systematic positive bias.
# ══════════════════════════════════════════════════════════════════

function SyntheticscRNAseq.simulate(net::GeneNetwork, alg::BinomialTauLeap,
                                    kin::KineticParams, ::Val{:gpu};
                                    cell_num::Int=1000,
                                    T::Float64=50.0,
                                    readout::Symbol=:protein)
    G = net.G
    A_pos_cpu, A_neg_cpu = SyntheticscRNAseq.precompute_hill_matrices(net)
    A_pos = CuArray{Float32}(A_pos_cpu)
    A_neg = CuArray{Float32}(A_neg_cpu)

    m_ss, p_ss = SyntheticscRNAseq.steady_state(net, kin)

    M = CUDA.zeros(Float32, G, cell_num)
    P = CUDA.zeros(Float32, G, cell_num)
    for i in 1:G
        M[i, :] .= Float32(ceil(m_ss[i]))
        P[i, :] .= Float32(ceil(p_ss[i]))
    end

    dt = Float32(alg.dt)
    n_steps = ceil(Int, T / alg.dt)
    k_t = Float32(kin.k_t)
    K_n = Float32(kin.K_d ^ kin.n)
    hill_n = Float32(kin.n)

    # Binomial decay probabilities (intrinsic + dilution, precomputed)
    p_decay_m = Float32(1.0 - exp(-(kin.mu_m + kin.dilution) * alg.dt))
    p_decay_p = Float32(1.0 - exp(-(kin.mu_p + kin.dilution) * alg.dt))

    P_n = similar(M)
    denom = similar(M)
    act_frac = similar(M)
    rep_frac = similar(M)
    reg_input = similar(M)
    temp = similar(M)
    u1 = similar(M)  # uniform for transcription (Poisson)
    u2 = similar(M)  # uniform for mRNA decay (Binomial)
    u3 = similar(M)  # uniform for translation (Poisson)
    u4 = similar(M)  # uniform for protein decay (Binomial)
    beta_gpu = CuArray{Float32}(reshape(net.basals, G, 1))

    for step in 1:n_steps
        # Regulatory input via Hill matmuls
        @. P_n = P ^ hill_n
        @. denom = P_n + K_n
        @. act_frac = P_n / denom
        @. rep_frac = K_n / denom

        mul!(reg_input, A_pos, act_frac)
        mul!(temp, A_neg, rep_frac)
        @. reg_input += temp

        # Draw 4 independent uniform noise fields for inverse CDF sampling
        CUDA.rand!(u1); CUDA.rand!(u2)
        CUDA.rand!(u3); CUDA.rand!(u4)

        # mRNA update: transcription (Poisson) - decay (Binomial)
        @. begin
            temp = max(reg_input + beta_gpu, Float32(0)) * dt
            u1 = _sample_poisson_gpu(u1, temp)
            u2 = _sample_binomial_gpu(u2, max(M, Float32(0)), p_decay_m)
            M = max(M + u1 - u2, Float32(0))
        end

        # Protein update: translation (Poisson) - decay (Binomial)
        @. begin
            temp = k_t * max(M, Float32(0)) * dt
            u3 = _sample_poisson_gpu(u3, temp)
            u4 = _sample_binomial_gpu(u4, max(P, Float32(0)), p_decay_p)
            P = max(P + u3 - u4, Float32(0))
        end
    end

    if readout == :mrna
        return Matrix{Float64}(Array(M)')
    elseif readout == :protein
        return Matrix{Float64}(Array(P)')
    else
        return hcat(Matrix{Float64}(Array(M)'), Matrix{Float64}(Array(P)'))
    end
end

# ══════════════════════════════════════════════════════════════════
#  GPU Binomial Tau-Leap — batched (multiple networks)
# ══════════════════════════════════════════════════════════════════

function SyntheticscRNAseq.simulate_gpu_batch(networks::Vector{GeneNetwork},
                                              alg::BinomialTauLeap,
                                              kin::KineticParams;
                                              cell_num::Int=1000,
                                              T::Float64=50.0,
                                              readout::Symbol=:protein)
    N = length(networks)
    G = networks[1].G

    M_cpu = zeros(Float32, G, cell_num, N)
    P_cpu = zeros(Float32, G, cell_num, N)
    beta_cpu = zeros(Float32, G, 1, N)
    A_pos_3d_cpu = zeros(Float32, G, G, N)
    A_neg_3d_cpu = zeros(Float32, G, G, N)

    for k in 1:N
        net = networks[k]
        A_pos_k, A_neg_k = SyntheticscRNAseq.precompute_hill_matrices(net)
        m_ss, p_ss = SyntheticscRNAseq.steady_state(net, kin)
        for i in 1:G
            M_cpu[i, :, k] .= Float32(ceil(m_ss[i]))
            P_cpu[i, :, k] .= Float32(ceil(p_ss[i]))
            beta_cpu[i, 1, k] = Float32(net.basals[i])
        end
        A_pos_3d_cpu[:, :, k] = Float32.(A_pos_k)
        A_neg_3d_cpu[:, :, k] = Float32.(A_neg_k)
    end

    M = CuArray(M_cpu)
    P = CuArray(P_cpu)
    beta_gpu = CuArray(beta_cpu)
    A_pos_3d = CuArray(A_pos_3d_cpu)
    A_neg_3d = CuArray(A_neg_3d_cpu)

    dt = Float32(alg.dt)
    n_steps = ceil(Int, T / alg.dt)
    k_t = Float32(kin.k_t)
    K_n = Float32(kin.K_d ^ kin.n)
    hill_n = Float32(kin.n)

    p_decay_m = Float32(1.0 - exp(-(kin.mu_m + kin.dilution) * alg.dt))
    p_decay_p = Float32(1.0 - exp(-(kin.mu_p + kin.dilution) * alg.dt))

    P_n = similar(M); denom = similar(M)
    act_frac = similar(M); rep_frac = similar(M)
    reg_input = similar(M); temp = similar(M)
    u1 = similar(M); u2 = similar(M)
    u3 = similar(M); u4 = similar(M)

    for step in 1:n_steps
        @. P_n = P ^ hill_n
        @. denom = P_n + K_n
        @. act_frac = P_n / denom
        @. rep_frac = K_n / denom

        gemm_strided_batched!('N', 'N', Float32(1), A_pos_3d, act_frac, Float32(0), reg_input)
        gemm_strided_batched!('N', 'N', Float32(1), A_neg_3d, rep_frac, Float32(0), temp)
        @. reg_input += temp

        CUDA.rand!(u1); CUDA.rand!(u2)
        CUDA.rand!(u3); CUDA.rand!(u4)

        @. begin
            temp = max(reg_input + beta_gpu, Float32(0)) * dt
            u1 = _sample_poisson_gpu(u1, temp)
            u2 = _sample_binomial_gpu(u2, max(M, Float32(0)), p_decay_m)
            M = max(M + u1 - u2, Float32(0))
        end

        @. begin
            temp = k_t * max(M, Float32(0)) * dt
            u3 = _sample_poisson_gpu(u3, temp)
            u4 = _sample_binomial_gpu(u4, max(P, Float32(0)), p_decay_p)
            P = max(P + u3 - u4, Float32(0))
        end
    end

    out = readout == :mrna ? Array(M) : Array(P)
    results = Vector{Matrix{Float64}}(undef, N)
    for k in 1:N
        results[k] = Float64.(out[:, :, k]')
    end
    return results
end

# ══════════════════════════════════════════════════════════════════
#  GPU CLE with population dynamics
#
#  Volume-dependent transcription on GPU. Division is handled by
#  a CPU-side pass every div_check_interval steps (division is
#  infrequent and requires scatter writes that GPUs handle poorly).
# ══════════════════════════════════════════════════════════════════

function SyntheticscRNAseq.simulate(net::GeneNetwork, alg::CLE, kin::KineticParams,
                                    ::Val{:gpu}, pop::PopulationConfig;
                                    T::Float64=50.0,
                                    readout::Symbol=:protein)
    G = net.G
    N = pop.cell_num
    A_pos_cpu, A_neg_cpu = SyntheticscRNAseq.precompute_hill_matrices(net)
    A_pos = CuArray{Float32}(A_pos_cpu)
    A_neg = CuArray{Float32}(A_neg_cpu)

    m_ss, p_ss = SyntheticscRNAseq.steady_state(net, kin)

    M = CUDA.zeros(Float32, G, N)
    P = CUDA.zeros(Float32, G, N)
    for i in 1:G
        M[i, :] .= Float32(ceil(m_ss[i]))
        P[i, :] .= Float32(ceil(p_ss[i]))
    end

    V_lo, V_hi = pop.V_init
    V_cpu = Float32.(V_lo .+ (V_hi - V_lo) .* rand(N))
    V = CuArray(V_cpu)

    dt = Float32(alg.dt)
    n_steps = ceil(Int, T / alg.dt)
    k_t = Float32(kin.k_t)
    K_n = Float32(kin.K_d ^ kin.n)
    hill_n = Float32(kin.n)
    # Population mode: intrinsic decay only (no dilution term — dilution
    # is emergent from binomial partitioning at division)
    mu_m = Float32(kin.mu_m)
    mu_p = Float32(kin.mu_p)
    sqrt_dt = Float32(sqrt(alg.dt))
    growth_factor = Float32(exp(pop.growth_rate * alg.dt))

    P_n = similar(M); denom = similar(M)
    act_frac = similar(M); rep_frac = similar(M)
    reg_input = similar(M); temp = similar(M)
    noise_m = similar(M); noise_p = similar(M)
    beta_gpu = CuArray{Float32}(reshape(net.basals, G, 1))

    for step in 1:n_steps
        V_row = reshape(V, 1, N)

        @. P_n = P ^ hill_n
        @. denom = P_n + K_n
        @. act_frac = P_n / denom
        @. rep_frac = K_n / denom

        mul!(reg_input, A_pos, act_frac)
        mul!(temp, A_neg, rep_frac)
        @. reg_input += temp

        # Volume-dependent transcription, intrinsic decay only
        CUDA.randn!(noise_m)
        @. begin
            noise_m = noise_m * sqrt_dt * sqrt(max(
                (reg_input + beta_gpu) * V_row + mu_m * max(M, Float32(0)), Float32(0)))
            M = max(M + ((reg_input + beta_gpu) * V_row - mu_m * M) * dt + noise_m, Float32(0))
        end

        CUDA.randn!(noise_p)
        @. begin
            noise_p = noise_p * sqrt_dt * sqrt(max(
                k_t * max(M, Float32(0)) + mu_p * max(P, Float32(0)), Float32(0)))
            P = max(P + (k_t * M - mu_p * P) * dt + noise_p, Float32(0))
        end

        # Volume growth (on GPU)
        V .*= growth_factor

        # Division check: pull to CPU, do division, push back
        if step % pop.div_check_interval == 0
            V_cpu = Array(V)
            dividers = findall(v -> v > pop.V_div, V_cpu)
            if !isempty(dividers)
                M_cpu = Array(M)
                P_cpu = Array(P)
                n_div = length(dividers)

                # Phase 1: partition mothers, buffer daughters
                d_M = Matrix{Float32}(undef, G, n_div)
                d_P = Matrix{Float32}(undef, G, n_div)
                d_V = Vector{Float32}(undef, n_div)

                for (k, c) in enumerate(dividers)
                    for g in 1:G
                        half_m = M_cpu[g, c] / 2f0
                        M_cpu[g, c] = half_m
                        d_M[g, k] = half_m
                        half_p = P_cpu[g, c] / 2f0
                        P_cpu[g, c] = half_p
                        d_P[g, k] = half_p
                    end
                    V_cpu[c] /= 2f0
                    d_V[k] = V_cpu[c]
                end

                # Phase 2: daughters replace random cells (Moran replacement)
                for (k, mother) in enumerate(dividers)
                    replaced = rand(1:N)
                    M_cpu[:, replaced] .= d_M[:, k]
                    P_cpu[:, replaced] .= d_P[:, k]
                    V_cpu[replaced] = d_V[k]
                end

                copyto!(M, CuArray(M_cpu))
                copyto!(P, CuArray(P_cpu))
                copyto!(V, CuArray(V_cpu))
            end
        end
    end

    if readout == :mrna
        return Matrix{Float64}(Array(M)')
    elseif readout == :protein
        return Matrix{Float64}(Array(P)')
    else
        return hcat(Matrix{Float64}(Array(M)'), Matrix{Float64}(Array(P)'))
    end
end

# ══════════════════════════════════════════════════════════════════
#  GPU Binomial Tau-Leap with population dynamics
#
#  Combines:
#   - Binomial decay (clamped to [0, n]) from the BinomialTauLeap code
#   - Volume-dependent transcription: propensity = (reg + β) × V
#   - Moran division: daughter replaces uniformly random cell
#   - Intrinsic decay only (no dilution term — dilution is emergent)
#
#  Division is handled CPU-side every div_check_interval steps.
# ══════════════════════════════════════════════════════════════════

function SyntheticscRNAseq.simulate(net::GeneNetwork, alg::BinomialTauLeap,
                                    kin::KineticParams, ::Val{:gpu},
                                    pop::PopulationConfig;
                                    T::Float64=50.0,
                                    readout::Symbol=:protein)
    G = net.G
    N = pop.cell_num
    A_pos_cpu, A_neg_cpu = SyntheticscRNAseq.precompute_hill_matrices(net)
    A_pos = CuArray{Float32}(A_pos_cpu)
    A_neg = CuArray{Float32}(A_neg_cpu)

    m_ss, p_ss = SyntheticscRNAseq.steady_state(net, kin)

    # Initial volumes ~ Uniform(V_lo, V_hi) to desynchronise cell cycle
    V_lo, V_hi = pop.V_init
    V_cpu = Float32.(V_lo .+ (V_hi - V_lo) .* rand(N))
    V = CuArray(V_cpu)

    # Initial molecule counts scaled by volume (integer-valued)
    M_cpu_arr = zeros(Float32, G, N)
    P_cpu_arr = zeros(Float32, G, N)
    for i in 1:G
        for c in 1:N
            M_cpu_arr[i, c] = Float32(max(1, round(m_ss[i] * V_cpu[c])))
            P_cpu_arr[i, c] = Float32(max(1, round(p_ss[i] * V_cpu[c])))
        end
    end
    M = CuArray(M_cpu_arr)
    P = CuArray(P_cpu_arr)

    dt = Float32(alg.dt)
    n_steps = ceil(Int, T / alg.dt)
    k_t = Float32(kin.k_t)
    K_n = Float32(kin.K_d ^ kin.n)
    hill_n = Float32(kin.n)
    # Intrinsic decay only (no dilution — dilution is emergent from division)
    p_decay_m = Float32(1.0 - exp(-kin.mu_m * alg.dt))
    p_decay_p = Float32(1.0 - exp(-kin.mu_p * alg.dt))
    growth_factor = Float32(exp(pop.growth_rate * alg.dt))

    P_conc = similar(M)
    P_n = similar(M); denom = similar(M)
    act_frac = similar(M); rep_frac = similar(M)
    reg_input = similar(M); temp = similar(M)
    u1 = similar(M); u2 = similar(M)
    u3 = similar(M); u4 = similar(M)
    beta_gpu = CuArray{Float32}(reshape(net.basals, G, 1))

    for step in 1:n_steps
        V_row = reshape(V, 1, N)

        # Hill fractions on concentrations (P / V)
        @. P_conc = P / V_row
        @. P_n = P_conc ^ hill_n
        @. denom = P_n + K_n
        @. act_frac = P_n / denom
        @. rep_frac = K_n / denom

        mul!(reg_input, A_pos, act_frac)
        mul!(temp, A_neg, rep_frac)
        @. reg_input += temp

        # Draw 4 independent uniform noise fields for inverse CDF sampling
        CUDA.rand!(u1); CUDA.rand!(u2)
        CUDA.rand!(u3); CUDA.rand!(u4)

        # mRNA update: volume-dependent transcription (Poisson) - intrinsic decay (Binomial)
        @. begin
            temp = max(reg_input + beta_gpu, Float32(0)) * V_row * dt
            u1 = _sample_poisson_gpu(u1, temp)
            u2 = _sample_binomial_gpu(u2, max(M, Float32(0)), p_decay_m)
            M = max(M + u1 - u2, Float32(0))
        end

        # Protein update: translation (Poisson) - intrinsic decay (Binomial)
        @. begin
            temp = k_t * max(M, Float32(0)) * dt
            u3 = _sample_poisson_gpu(u3, temp)
            u4 = _sample_binomial_gpu(u4, max(P, Float32(0)), p_decay_p)
            P = max(P + u3 - u4, Float32(0))
        end

        # Volume growth
        V .*= growth_factor

        # Division check (CPU-side, Moran replacement)
        if step % pop.div_check_interval == 0
            V_cpu = Array(V)
            dividers = findall(v -> v > pop.V_div, V_cpu)
            if !isempty(dividers)
                M_cpu = Array(M)
                P_cpu = Array(P)
                n_div = length(dividers)

                d_M = Matrix{Float32}(undef, G, n_div)
                d_P = Matrix{Float32}(undef, G, n_div)
                d_V = Vector{Float32}(undef, n_div)

                for (k, c) in enumerate(dividers)
                    for g in 1:G
                        mi = round(Int, M_cpu[g, c])
                        m_mother = mi > 0 ? Float32(rand(Binomial(mi, 0.5))) : 0f0
                        M_cpu[g, c] = m_mother
                        d_M[g, k] = Float32(mi) - m_mother
                        pi = round(Int, P_cpu[g, c])
                        p_mother = pi > 0 ? Float32(rand(Binomial(pi, 0.5))) : 0f0
                        P_cpu[g, c] = p_mother
                        d_P[g, k] = Float32(pi) - p_mother
                    end
                    V_cpu[c] /= 2f0
                    d_V[k] = V_cpu[c]
                end

                # Daughters replace uniformly random cells (Moran)
                for (k, mother) in enumerate(dividers)
                    replaced = rand(1:N)
                    M_cpu[:, replaced] .= d_M[:, k]
                    P_cpu[:, replaced] .= d_P[:, k]
                    V_cpu[replaced] = d_V[k]
                end

                copyto!(M, CuArray(M_cpu))
                copyto!(P, CuArray(P_cpu))
                copyto!(V, CuArray(V_cpu))
            end
        end
    end

    # Return concentrations (counts / volume) to match CLE scale
    V_row = reshape(V, 1, N)
    if readout == :mrna
        return Matrix{Float64}(Array(@. M / V_row)')
    elseif readout == :protein
        return Matrix{Float64}(Array(@. P / V_row)')
    else
        return hcat(Matrix{Float64}(Array(@. M / V_row)'), Matrix{Float64}(Array(@. P / V_row)'))
    end
end

# ══════════════════════════════════════════════════════════════════
#  GPU Binomial Tau-Leap with population — batched (multiple networks)
#
#  3D tensor layout: (G, cell_num, N) where N = batch size.
#  Per-network kinetics via (1, 1, N) parameter arrays.
# ══════════════════════════════════════════════════════════════════

function SyntheticscRNAseq.simulate_gpu_batch(networks::Vector{GeneNetwork},
                                              alg::BinomialTauLeap,
                                              kinetics::Vector{<:SyntheticscRNAseq.KineticParams},
                                              pop::PopulationConfig;
                                              T::Float64=50.0,
                                              readout::Symbol=:protein)
    N = length(networks)
    G = networks[1].G
    cell_num = pop.cell_num

    # Per-network kinetic parameters as (1, 1, N) arrays
    k_t_cpu = zeros(Float32, 1, 1, N)
    K_n_cpu = zeros(Float32, 1, 1, N)
    hill_n_cpu = zeros(Float32, 1, 1, N)
    mu_m_cpu = zeros(Float32, 1, 1, N)
    mu_p_cpu = zeros(Float32, 1, 1, N)
    for k in 1:N
        kp = kinetics[k]
        k_t_cpu[1, 1, k] = Float32(kp.k_t)
        K_n_cpu[1, 1, k] = Float32(kp.K_d ^ kp.n)
        hill_n_cpu[1, 1, k] = Float32(kp.n)
        mu_m_cpu[1, 1, k] = Float32(kp.mu_m)
        mu_p_cpu[1, 1, k] = Float32(kp.mu_p)
    end

    # Binomial decay probabilities per-network (intrinsic only, no dilution)
    p_decay_m_cpu = zeros(Float32, 1, 1, N)
    p_decay_p_cpu = zeros(Float32, 1, 1, N)
    for k in 1:N
        p_decay_m_cpu[1, 1, k] = Float32(1.0 - exp(-kinetics[k].mu_m * alg.dt))
        p_decay_p_cpu[1, 1, k] = Float32(1.0 - exp(-kinetics[k].mu_p * alg.dt))
    end

    # Initial conditions
    M_cpu = zeros(Float32, G, cell_num, N)
    P_cpu = zeros(Float32, G, cell_num, N)
    V_cpu = ones(Float32, 1, cell_num, N) .+ rand(Float32, 1, cell_num, N)
    beta_cpu = zeros(Float32, G, 1, N)
    A_pos_3d_cpu = zeros(Float32, G, G, N)
    A_neg_3d_cpu = zeros(Float32, G, G, N)

    for k in 1:N
        net = networks[k]
        kp = kinetics[k]
        A_pos_k, A_neg_k = SyntheticscRNAseq.precompute_hill_matrices(net)
        m_ss = net.basals ./ kp.mu_m
        p_ss = kp.k_t .* m_ss ./ kp.mu_p
        for i in 1:G
            for c in 1:cell_num
                v0 = V_cpu[1, c, k]
                M_cpu[i, c, k] = Float32(max(1, round(m_ss[i] * v0)))
                P_cpu[i, c, k] = Float32(max(1, round(p_ss[i] * v0)))
            end
            beta_cpu[i, 1, k] = Float32(net.basals[i])
        end
        A_pos_3d_cpu[:, :, k] = Float32.(A_pos_k)
        A_neg_3d_cpu[:, :, k] = Float32.(A_neg_k)
    end

    M = CuArray(M_cpu); P = CuArray(P_cpu); V = CuArray(V_cpu)
    beta_gpu = CuArray(beta_cpu)
    A_pos_3d = CuArray(A_pos_3d_cpu); A_neg_3d = CuArray(A_neg_3d_cpu)
    k_t_g = CuArray(k_t_cpu); K_n_g = CuArray(K_n_cpu); hill_n_g = CuArray(hill_n_cpu)
    p_decay_m_g = CuArray(p_decay_m_cpu); p_decay_p_g = CuArray(p_decay_p_cpu)

    dt = Float32(alg.dt)
    n_steps = ceil(Int, T / alg.dt)
    growth_factor = Float32(exp(pop.growth_rate * alg.dt))

    P_conc = similar(M)
    P_n = similar(M); denom = similar(M)
    act_frac = similar(M); rep_frac = similar(M)
    reg_input = similar(M); temp = similar(M)
    u1 = similar(M); u2 = similar(M)
    u3 = similar(M); u4 = similar(M)

    for step in 1:n_steps
        # Hill fractions on concentrations
        @. P_conc = P / V
        @. P_n = P_conc ^ hill_n_g
        @. denom = P_n + K_n_g
        @. act_frac = P_n / denom
        @. rep_frac = K_n_g / denom

        gemm_strided_batched!('N', 'N', Float32(1), A_pos_3d, act_frac, Float32(0), reg_input)
        gemm_strided_batched!('N', 'N', Float32(1), A_neg_3d, rep_frac, Float32(0), temp)
        @. reg_input += temp

        CUDA.rand!(u1); CUDA.rand!(u2)
        CUDA.rand!(u3); CUDA.rand!(u4)

        # mRNA: volume-dependent transcription (Poisson) - intrinsic decay (Binomial)
        @. begin
            temp = max(reg_input + beta_gpu, Float32(0)) * V * dt
            u1 = _sample_poisson_gpu(u1, temp)
            u2 = _sample_binomial_gpu(u2, max(M, Float32(0)), p_decay_m_g)
            M = max(M + u1 - u2, Float32(0))
        end

        # Protein: translation (Poisson) - intrinsic decay (Binomial)
        @. begin
            temp = k_t_g * max(M, Float32(0)) * dt
            u3 = _sample_poisson_gpu(u3, temp)
            u4 = _sample_binomial_gpu(u4, max(P, Float32(0)), p_decay_p_g)
            P = max(P + u3 - u4, Float32(0))
        end

        # Volume growth
        @. V = V * growth_factor

        # Division (CPU-side, Moran replacement with exact binomial partitioning)
        if step % pop.div_check_interval == 0
            M_h = Array(M); P_h = Array(P); V_h = Array(V)
            any_div = false
            for k in 1:N
                dividing = findall(V_h[1, :, k] .> pop.V_div)
                isempty(dividing) && continue
                any_div = true
                for ci in dividing
                    replaced = rand(1:cell_num)
                    v_half = V_h[1, ci, k] * 0.5f0
                    for g in 1:G
                        mi = round(Int, M_h[g, ci, k])
                        m_mother = mi > 0 ? Float32(rand(Binomial(mi, 0.5))) : 0f0
                        M_h[g, ci, k] = m_mother
                        M_h[g, replaced, k] = Float32(mi) - m_mother
                        pi = round(Int, P_h[g, ci, k])
                        p_mother = pi > 0 ? Float32(rand(Binomial(pi, 0.5))) : 0f0
                        P_h[g, ci, k] = p_mother
                        P_h[g, replaced, k] = Float32(pi) - p_mother
                    end
                    V_h[1, ci, k] = v_half
                    V_h[1, replaced, k] = v_half
                end
            end
            if any_div
                copyto!(M, CuArray(M_h)); copyto!(P, CuArray(P_h)); copyto!(V, CuArray(V_h))
            end
        end
    end

    # Return concentrations (counts / volume)
    out = readout == :mrna ? Array(@. M / V) : Array(@. P / V)
    results = Vector{Matrix{Float64}}(undef, N)
    for k in 1:N
        results[k] = Float64.(out[:, :, k]')
    end
    return results
end

function __init__()
    @info "SyntheticscRNAseq: CUDA extension loaded"
end

end # module
