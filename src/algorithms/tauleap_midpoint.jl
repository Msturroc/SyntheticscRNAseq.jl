#= ================================================================
   Midpoint tau-leap — O(dt^2) bias reduction.

   1. Deterministic half-step predictor: X_mid = X + (dt/2) * drift(X)
   2. Full step with propensities evaluated at X_mid

   This reduces the systematic bias from O(dt) to O(dt^2) at the
   same computational cost per step.

   Uses Poisson events at midpoint propensities (could also use
   Binomial — future enhancement).
   ================================================================ =#

"""
    MidpointTauLeap(dt)

Midpoint tau-leap algorithm with O(dt^2) bias.
"""
struct MidpointTauLeap
    dt::Float64
end

"""
    simulate(network, alg::MidpointTauLeap, kinetics; cell_num, T, readout, rng)

Midpoint tau-leap simulation. Same interface as SSA.
"""
function simulate(net::GeneNetwork, alg::MidpointTauLeap, kin::KineticParams;
                  cell_num::Int=1000,
                  T::Float64=50.0,
                  readout::Symbol=:protein,
                  rng::AbstractRNG=Random.default_rng(),
                  population::Union{Nothing, PopulationConfig}=nothing)

    if population !== nothing
        error("MidpointTauLeap with population not implemented. Use BinomialTauLeap.")
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
            # ── Half-step predictor (deterministic) ──
            m_mid = copy(m)
            p_mid = copy(p)

            for i in 1:G
                reg = regulatory_input(net, Float64.(p), i, kin.K_d, kin.n)
                prop_transcription = max(reg + net.basals[i], 0.0)

                drift_m = prop_transcription - (kin.mu_m + kin.dilution) * m[i]
                drift_p = kin.k_t * m[i] - (kin.mu_p + kin.dilution) * p[i]

                m_mid[i] = max(round(Int, m[i] + 0.5 * dt * drift_m), 0)
                p_mid[i] = max(round(Int, p[i] + 0.5 * dt * drift_p), 0)
            end

            # ── Full step with midpoint propensities ──
            for i in 1:G
                reg_mid = regulatory_input(net, Float64.(p_mid), i, kin.K_d, kin.n)
                prop_transcription_mid = max(reg_mid + net.basals[i], 0.0)

                n_transcribe = rand(rng, Distributions.Poisson(prop_transcription_mid * dt))
                n_mrna_decay = rand(rng, Distributions.Poisson((kin.mu_m + kin.dilution) * max(m_mid[i], 0) * dt))
                n_translate  = rand(rng, Distributions.Poisson(kin.k_t * max(m_mid[i], 0) * dt))
                n_prot_decay = rand(rng, Distributions.Poisson((kin.mu_p + kin.dilution) * max(p_mid[i], 0) * dt))

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
