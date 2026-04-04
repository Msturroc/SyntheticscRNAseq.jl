module SyntheticscRNAseqCUDAExt

using SyntheticscRNAseq
using CUDA
using CUDA.CUBLAS: gemm_strided_batched!
using LinearAlgebra
using Random

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
    z1 = similar(M)  # noise for transcription (Poisson approx)
    z2 = similar(M)  # noise for mRNA decay (Binomial approx)
    z3 = similar(M)  # noise for translation (Poisson approx)
    z4 = similar(M)  # noise for protein decay (Binomial approx)
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

        # Draw 4 independent noise fields
        CUDA.randn!(z1)
        CUDA.randn!(z2)
        CUDA.randn!(z3)
        CUDA.randn!(z4)

        # mRNA update: transcription (Poisson) - decay (Binomial)
        # Transcription: λ = (reg + β) * dt
        #   approx: max(λ + sqrt(λ) * z, 0)  (Poisson normal approx)
        # Decay: Binomial(M, p_decay_m)
        #   approx: clamp(M*p + sqrt(M*p*(1-p)) * z, 0, M)
        @. begin
            # Transcription (Poisson normal approx, clamped ≥ 0)
            temp = max(reg_input + beta_gpu, Float32(0)) * dt
            z1 = max(temp + sqrt(max(temp, Float32(0))) * z1, Float32(0))

            # mRNA decay (Binomial normal approx, clamped to [0, M])
            z2 = min(max(
                max(M, Float32(0)) * p_decay_m +
                sqrt(max(M, Float32(0)) * p_decay_m * (Float32(1) - p_decay_m)) * z2,
                Float32(0)), max(M, Float32(0)))

            M = max(M + z1 - z2, Float32(0))
        end

        # Protein update: translation (Poisson) - decay (Binomial)
        @. begin
            # Translation (Poisson normal approx)
            temp = k_t * max(M, Float32(0)) * dt
            z3 = max(temp + sqrt(max(temp, Float32(0))) * z3, Float32(0))

            # Protein decay (Binomial normal approx)
            z4 = min(max(
                max(P, Float32(0)) * p_decay_p +
                sqrt(max(P, Float32(0)) * p_decay_p * (Float32(1) - p_decay_p)) * z4,
                Float32(0)), max(P, Float32(0)))

            P = max(P + z3 - z4, Float32(0))
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
    z1 = similar(M); z2 = similar(M)
    z3 = similar(M); z4 = similar(M)

    for step in 1:n_steps
        @. P_n = P ^ hill_n
        @. denom = P_n + K_n
        @. act_frac = P_n / denom
        @. rep_frac = K_n / denom

        gemm_strided_batched!('N', 'N', Float32(1), A_pos_3d, act_frac, Float32(0), reg_input)
        gemm_strided_batched!('N', 'N', Float32(1), A_neg_3d, rep_frac, Float32(0), temp)
        @. reg_input += temp

        CUDA.randn!(z1); CUDA.randn!(z2)
        CUDA.randn!(z3); CUDA.randn!(z4)

        @. begin
            temp = max(reg_input + beta_gpu, Float32(0)) * dt
            z1 = max(temp + sqrt(max(temp, Float32(0))) * z1, Float32(0))
            z2 = min(max(
                max(M, Float32(0)) * p_decay_m +
                sqrt(max(M, Float32(0)) * p_decay_m * (Float32(1) - p_decay_m)) * z2,
                Float32(0)), max(M, Float32(0)))
            M = max(M + z1 - z2, Float32(0))
        end

        @. begin
            temp = k_t * max(M, Float32(0)) * dt
            z3 = max(temp + sqrt(max(temp, Float32(0))) * z3, Float32(0))
            z4 = min(max(
                max(P, Float32(0)) * p_decay_p +
                sqrt(max(P, Float32(0)) * p_decay_p * (Float32(1) - p_decay_p)) * z4,
                Float32(0)), max(P, Float32(0)))
            P = max(P + z3 - z4, Float32(0))
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

                # Phase 2: daughters replace random cells
                for (k, mother) in enumerate(dividers)
                    replaced = rand(1:N-1)
                    if replaced >= mother
                        replaced += 1
                    end
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

function __init__()
    @info "SyntheticscRNAseq: CUDA extension loaded"
end

end # module
