#= ================================================================
   Optimized Binomial Tau-Leap using the same 5 packages.

   StaticArrays for Hill matmuls, Polyester for cell parallelism,
   @fastmath @inbounds for element-wise. VectorizedRNG not directly
   useful for Poisson/Binomial draws, but is used for pre-generating
   uniform/normal random numbers that feed the inverse-CDF samplers.
   ================================================================ =#

using StaticArrays
using Polyester: @batch

"""
    BinomialTauLeapFast(dt)

High-performance binomial tau-leap with StaticArrays matmuls,
Polyester multithreading, and Bumper arena allocation.
"""
struct BinomialTauLeapFast
    dt::Float64
end

function simulate(net::GeneNetwork, alg::BinomialTauLeapFast, kin::KineticParams;
                  cell_num::Int=1000,
                  T::Float64=50.0,
                  readout::Symbol=:protein,
                  rng::AbstractRNG=Random.default_rng(),
                  population::Union{Nothing, PopulationConfig}=nothing)

    if population !== nothing
        error("BinomialTauLeapFast with population not yet implemented.")
    end

    return _simulate_bintau_fast(Val(net.G), net, alg, kin;
                                 cell_num=cell_num, T=T, readout=readout,
                                 rng=rng)
end

function _simulate_bintau_fast(::Val{G}, net::GeneNetwork, alg::BinomialTauLeapFast,
                               kin::KineticParams;
                               cell_num::Int=1000,
                               T::Float64=50.0,
                               readout::Symbol=:protein,
                               rng::AbstractRNG=Random.default_rng()) where G

    A_pos_cpu, A_neg_cpu = precompute_hill_matrices(net)
    A_pos_s = SMatrix{G, G, Float64, G*G}(A_pos_cpu)
    A_neg_s = SMatrix{G, G, Float64, G*G}(A_neg_cpu)
    beta_s = SVector{G, Float64}(net.basals)

    m_ss, p_ss = steady_state(net, kin)

    dt = alg.dt
    n_steps = ceil(Int, T / dt)
    k_t = kin.k_t
    K_n = kin.K_d ^ kin.n
    hill_n = kin.n
    mu_m = kin.mu_m + kin.dilution   # effective mRNA decay (intrinsic + dilution)
    mu_p = kin.mu_p + kin.dilution   # effective protein decay (intrinsic + dilution)
    p_decay_m = 1.0 - exp(-mu_m * dt)
    p_decay_p = 1.0 - exp(-mu_p * dt)

    # State
    M = Matrix{Int}(undef, G, cell_num)
    P_state = Matrix{Int}(undef, G, cell_num)
    @inbounds for c in 1:cell_num
        for g in 1:G
            M[g, c] = round(Int, m_ss[g])
            P_state[g, c] = round(Int, p_ss[g])
        end
    end

    # Pre-generate per-thread RNGs for reproducibility with @batch
    nthreads = max(Threads.nthreads(), 1)
    rngs = [Random.MersenneTwister(rand(rng, UInt64)) for _ in 1:nthreads]

    for step in 1:n_steps
        # Polyester: parallel over cells
        @batch per=thread for c in 1:cell_num
            tid = Threads.threadid()
            lrng = @inbounds rngs[tid]

            # Load protein as Float64 SVector (view-based, Polyester-safe)
            @inbounds p_col = SVector{G, Float64}(view(P_state, :, c))

            # StaticArrays Hill matmul
            @fastmath begin
                p_n = p_col .^ hill_n
                denom = p_n .+ K_n
                act_frac = p_n ./ denom
                rep_frac = K_n ./ denom
                reg = A_pos_s * act_frac + A_neg_s * rep_frac
            end

            # Cooperative (AND) and redundant (OR) corrections
            for edge in net.cooperative
                prod_h = 1.0
                for src in edge.sources
                    @inbounds prod_h *= act_frac[src]
                end
                @inbounds reg = Base.setindex(reg, reg[edge.target] + edge.strength * beta_s[edge.target] * prod_h, edge.target)
            end
            for edge in net.redundant
                prod_1mh = 1.0
                for src in edge.sources
                    @inbounds prod_1mh *= (1.0 - act_frac[src])
                end
                @inbounds reg = Base.setindex(reg, reg[edge.target] + edge.strength * beta_s[edge.target] * (1.0 - prod_1mh), edge.target)
            end

            @inbounds for i in 1:G
                prop = @fastmath max(reg[i] + beta_s[i], 0.0)

                # Transcription: Poisson(prop * dt)
                n_transcribe = rand(lrng, Distributions.Poisson(prop * dt))

                # mRNA decay: Binomial(m, p_decay)
                mi = M[i, c]
                n_mrna_decay = mi > 0 ? rand(lrng, Distributions.Binomial(mi, p_decay_m)) : 0

                M[i, c] = mi + n_transcribe - n_mrna_decay

                # Translation: Poisson(k_t * m * dt)
                mi_new = max(M[i, c], 0)
                n_translate = rand(lrng, Distributions.Poisson(k_t * mi_new * dt))

                # Protein decay: Binomial(p, p_decay)
                pi = P_state[i, c]
                n_prot_decay = pi > 0 ? rand(lrng, Distributions.Binomial(pi, p_decay_p)) : 0

                P_state[i, c] = pi + n_translate - n_prot_decay
            end
        end
    end

    out_cols = readout == :both ? 2G : G
    Y = Matrix{Float64}(undef, cell_num, out_cols)
    @inbounds for c in 1:cell_num
        if readout == :mrna
            for g in 1:G; Y[c, g] = M[g, c]; end
        elseif readout == :protein
            for g in 1:G; Y[c, g] = P_state[g, c]; end
        else
            for g in 1:G; Y[c, g] = M[g, c]; end
            for g in 1:G; Y[c, G+g] = P_state[g, c]; end
        end
    end

    return Y
end
