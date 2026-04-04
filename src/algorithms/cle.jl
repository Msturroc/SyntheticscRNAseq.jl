#= ================================================================
   Chemical Langevin Equation (CLE) — CPU vectorized.

   Continuous approximation of the CME. Valid when molecule counts
   are large enough that Poisson → Gaussian is reasonable.

   Ported from GeneticNetworkSBI/src/cle_vectorized.jl (validated
   to < 0.7% error vs Catalyst.jl SSA).

   Uses collapsed noise model: sum of independent Gaussian noise
   sources per gene reduces to a single Gaussian draw per gene.
   ================================================================ =#

using LinearAlgebra

"""
    CLE(dt)

Chemical Langevin Equation with Euler-Maruyama and collapsed noise.
"""
struct CLE
    dt::Float64
end

"""
    simulate(network, alg::CLE, kinetics; cell_num, T, readout, rng, population)

Vectorized CLE simulation. All cells evolve simultaneously via
matrix operations.

When `population` is provided, simulates with Moran dynamics and
volume-dependent transcription (continuous approximation).
"""
function simulate(net::GeneNetwork, alg::CLE, kin::KineticParams;
                  cell_num::Int=1000,
                  T::Float64=50.0,
                  readout::Symbol=:protein,
                  rng::AbstractRNG=Random.default_rng(),
                  population::Union{Nothing, PopulationConfig}=nothing)

    if population !== nothing
        return _simulate_cle_population(net, alg, kin, population;
                                        T=T, readout=readout, rng=rng)
    end

    G = net.G
    A_pos, A_neg = precompute_hill_matrices(net)

    m_ss, p_ss = steady_state(net, kin)

    # State matrices: (G x cell_num) for column-major matmul efficiency
    M = repeat(Float64.(ceil.(m_ss)), 1, cell_num)
    P = repeat(Float64.(ceil.(p_ss)), 1, cell_num)

    dt = alg.dt
    n_steps = ceil(Int, T / dt)
    k_t = kin.k_t
    K_n = kin.K_d ^ kin.n
    hill_n = kin.n
    mu_m_eff = kin.mu_m + kin.dilution  # effective mRNA decay (intrinsic + dilution)
    mu_p_eff = kin.mu_p + kin.dilution  # effective protein decay (intrinsic + dilution)
    sqrt_dt = sqrt(dt)

    # Temporaries
    P_n = similar(P)
    denom = similar(P)
    act_frac = similar(P)
    rep_frac = similar(P)
    reg_input = similar(M)
    noise_m = similar(M)
    noise_p = similar(P)

    beta_vec = reshape(net.basals, G, 1)

    has_coop = !isempty(net.cooperative)
    has_redun = !isempty(net.redundant)

    for step in 1:n_steps
        @. P_n = P ^ hill_n
        @. denom = P_n + K_n
        @. act_frac = P_n / denom
        @. rep_frac = K_n / denom

        mul!(reg_input, A_pos, act_frac)
        mul!(noise_m, A_neg, rep_frac)  # reuse as temp
        @. reg_input += noise_m

        # Cooperative (AND) and redundant (OR) corrections
        if has_coop || has_redun
            @inbounds for c in 1:cell_num
                for edge in net.cooperative
                    prod_h = 1.0
                    for src in edge.sources
                        prod_h *= act_frac[src, c]
                    end
                    reg_input[edge.target, c] += edge.strength * net.basals[edge.target] * prod_h
                end
                for edge in net.redundant
                    prod_1mh = 1.0
                    for src in edge.sources
                        prod_1mh *= (1.0 - act_frac[src, c])
                    end
                    reg_input[edge.target, c] += edge.strength * net.basals[edge.target] * (1.0 - prod_1mh)
                end
            end
        end

        # mRNA update (decay includes dilution)
        randn!(rng, noise_m)
        @. begin
            noise_m = noise_m * sqrt_dt * sqrt(max(reg_input + beta_vec + mu_m_eff * max(M, 0.0), 0.0))
            M = max(M + (reg_input + beta_vec - mu_m_eff * M) * dt + noise_m, 0.0)
        end

        # Protein update (decay includes dilution)
        randn!(rng, noise_p)
        @. begin
            noise_p = noise_p * sqrt_dt * sqrt(max(k_t * max(M, 0.0) + mu_p_eff * max(P, 0.0), 0.0))
            P = max(P + (k_t * M - mu_p_eff * P) * dt + noise_p, 0.0)
        end
    end

    if readout == :mrna
        return Matrix(M')
    elseif readout == :protein
        return Matrix(P')
    else  # :both
        return hcat(Matrix(M'), Matrix(P'))
    end
end

# ── CLE with population dynamics ─────────────────────────────────

"""
    _simulate_cle_population(net, alg, kin, pop; T, readout, rng)

CLE with Moran population dynamics. Volume-dependent transcription,
no dilution term, exponential growth, binomial division.

Uses per-cell loop (not fully vectorized) due to division events
changing cell states discontinuously.
"""
function _simulate_cle_population(net::GeneNetwork, alg::CLE,
                                  kin::KineticParams, pop::PopulationConfig;
                                  T::Float64=50.0,
                                  readout::Symbol=:protein,
                                  rng::AbstractRNG=Random.default_rng())
    G = net.G
    N = pop.cell_num
    A_pos, A_neg = precompute_hill_matrices(net)

    m_ss, p_ss = steady_state(net, kin)

    # State: (G, N) continuous concentrations + (N,) volumes
    M = repeat(Float64.(ceil.(m_ss)), 1, N)
    P = repeat(Float64.(ceil.(p_ss)), 1, N)

    V_lo, V_hi = pop.V_init
    V = V_lo .+ (V_hi - V_lo) .* rand(rng, N)

    dt = alg.dt
    n_steps = ceil(Int, T / dt)
    k_t = kin.k_t
    K_n = kin.K_d ^ kin.n
    hill_n = kin.n
    mu_m = kin.mu_m
    mu_p = kin.mu_p
    sqrt_dt = sqrt(dt)
    growth_factor = exp(pop.growth_rate * dt)

    # Temporaries
    P_n = similar(P)
    denom = similar(P)
    act_frac = similar(P)
    rep_frac = similar(P)
    reg_input = similar(M)
    noise_m = similar(M)
    noise_p = similar(P)

    beta_vec = reshape(net.basals, G, 1)

    has_coop = !isempty(net.cooperative)
    has_redun = !isempty(net.redundant)

    for step in 1:n_steps
        # Volume as (1, N) for broadcasting
        V_row = reshape(V, 1, N)

        # Hill function computation (vectorized)
        @. P_n = P ^ hill_n
        @. denom = P_n + K_n
        @. act_frac = P_n / denom
        @. rep_frac = K_n / denom

        mul!(reg_input, A_pos, act_frac)
        mul!(noise_m, A_neg, rep_frac)
        @. reg_input += noise_m

        # Cooperative (AND) and redundant (OR) corrections
        if has_coop || has_redun
            @inbounds for c in 1:N
                for edge in net.cooperative
                    prod_h = 1.0
                    for src in edge.sources
                        prod_h *= act_frac[src, c]
                    end
                    reg_input[edge.target, c] += edge.strength * net.basals[edge.target] * prod_h
                end
                for edge in net.redundant
                    prod_1mh = 1.0
                    for src in edge.sources
                        prod_1mh *= (1.0 - act_frac[src, c])
                    end
                    reg_input[edge.target, c] += edge.strength * net.basals[edge.target] * (1.0 - prod_1mh)
                end
            end
        end

        # mRNA: volume-dependent transcription, intrinsic decay only
        # drift = (reg + β) * V - μ_m * m
        randn!(rng, noise_m)
        @. begin
            noise_m = noise_m * sqrt_dt * sqrt(max(
                (reg_input + beta_vec) * V_row + mu_m * max(M, 0.0), 0.0))
            M = max(M + ((reg_input + beta_vec) * V_row - mu_m * M) * dt + noise_m, 0.0)
        end

        # Protein: intrinsic decay only
        randn!(rng, noise_p)
        @. begin
            noise_p = noise_p * sqrt_dt * sqrt(max(
                k_t * max(M, 0.0) + mu_p * max(P, 0.0), 0.0))
            P = max(P + (k_t * M - mu_p * P) * dt + noise_p, 0.0)
        end

        # Volume growth
        V .*= growth_factor

        # Division check (two-phase: partition first, then swap in)
        if step % pop.div_check_interval == 0
            dividers = findall(v -> v > pop.V_div, V)
            if !isempty(dividers)
                # Phase 1: halve mothers, store daughter halves in buffer
                n_div = length(dividers)
                d_M = Matrix{Float64}(undef, G, n_div)
                d_P = Matrix{Float64}(undef, G, n_div)
                d_V = Vector{Float64}(undef, n_div)

                for (k, c) in enumerate(dividers)
                    for g in 1:G
                        half_m = M[g, c] / 2.0
                        M[g, c] = half_m
                        d_M[g, k] = half_m

                        half_p = P[g, c] / 2.0
                        P[g, c] = half_p
                        d_P[g, k] = half_p
                    end
                    V[c] /= 2.0
                    d_V[k] = V[c]
                end

                # Phase 2: swap daughters into random slots
                for (k, mother) in enumerate(dividers)
                    replaced = rand(rng, 1:N-1)
                    if replaced >= mother
                        replaced += 1
                    end
                    M[:, replaced] .= d_M[:, k]
                    P[:, replaced] .= d_P[:, k]
                    V[replaced] = d_V[k]
                end
            end
        end
    end

    if readout == :mrna
        return Matrix(M')
    elseif readout == :protein
        return Matrix(P')
    else
        return hcat(Matrix(M'), Matrix(P'))
    end
end
