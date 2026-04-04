#= ================================================================
   Fixed-step Poisson tau-leap (CPU).

   Simplest tau-leap: at each step, draw Poisson(propensity * dt)
   events for each reaction. Known to have O(dt) bias and can go
   negative for small counts.

   Primarily useful as a reference to compare against the binomial
   tau-leap (which is naturally non-negative).
   ================================================================ =#

"""
    PoissonTauLeap(dt)

Fixed-step Poisson tau-leap algorithm.
"""
struct PoissonTauLeap
    dt::Float64
end

"""
    simulate(network, alg::PoissonTauLeap, kinetics; cell_num, T, readout, rng)

Poisson tau-leap simulation. Same interface as SSA.
"""
function simulate(net::GeneNetwork, alg::PoissonTauLeap, kin::KineticParams;
                  cell_num::Int=1000,
                  T::Float64=50.0,
                  readout::Symbol=:protein,
                  rng::AbstractRNG=Random.default_rng(),
                  population::Union{Nothing, PopulationConfig}=nothing)

    if population !== nothing
        error("PoissonTauLeap with population not implemented. Use BinomialTauLeap.")
    end
    G = net.G
    dt = alg.dt
    n_steps = ceil(Int, T / dt)
    out_cols = readout == :both ? 2G : G

    m_ss, p_ss = steady_state(net, kin)
    Y = Matrix{Float64}(undef, cell_num, out_cols)

    for cell in 1:cell_num
        m = round.(Int, m_ss)
        p = round.(Int, p_ss)

        for step in 1:n_steps
            for i in 1:G
                reg = regulatory_input(net, Float64.(p), i, kin.K_d, kin.n)
                prop_transcription = max(reg + net.basals[i], 0.0)

                # Poisson draws for each reaction
                n_transcribe = rand(rng, Distributions.Poisson(prop_transcription * dt))
                n_mrna_decay = rand(rng, Distributions.Poisson((kin.mu_m + kin.dilution) * max(m[i], 0) * dt))
                n_translate  = rand(rng, Distributions.Poisson(kin.k_t * max(m[i], 0) * dt))
                n_prot_decay = rand(rng, Distributions.Poisson((kin.mu_p + kin.dilution) * max(p[i], 0) * dt))

                m[i] = max(m[i] + n_transcribe - n_mrna_decay, 0)
                p[i] = max(p[i] + n_translate - n_prot_decay, 0)
            end
        end

        if readout == :mrna
            Y[cell, :] .= m
        elseif readout == :protein
            Y[cell, :] .= p
        else
            Y[cell, 1:G] .= m
            Y[cell, G+1:2G] .= p
        end
    end

    return Y
end
