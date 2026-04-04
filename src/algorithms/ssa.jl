#= ================================================================
   Gillespie Stochastic Simulation Algorithm (SSA) — exact.

   Direct method: at each step, compute all propensities, draw
   exponential waiting time, choose reaction proportional to
   propensity, update state.

   Species layout for G genes:
     indices 1:G     = mRNA counts (integers)
     indices G+1:2G  = protein counts (integers)

   Reactions for gene i:
     1. Transcription:  ∅ → mRNA_i        rate = reg_input_i + β_i
     2. mRNA decay:     mRNA_i → ∅        rate = μ_m * mRNA_i
     3. Translation:    mRNA_i → mRNA_i + protein_i    rate = k_t * mRNA_i
     4. Protein decay:  protein_i → ∅     rate = μ_p * protein_i

   Total reactions: 4G
   ================================================================ =#

"""
    SSA

Algorithm type for exact Gillespie Stochastic Simulation Algorithm.
"""
struct SSA end

"""
    simulate(network, ::SSA, kinetics; cell_num, T, readout, rng)

Run exact SSA for a gene regulatory network. Simulates `cell_num`
independent cells to time `T` and returns a snapshot matrix.

Returns:
- Y: Matrix (cell_num x G) of molecule counts at time T
  (:mrna for mRNA, :protein for protein, :both for 2G columns)
"""
function simulate(net::GeneNetwork, ::SSA, kin::KineticParams;
                  cell_num::Int=1000,
                  T::Float64=50.0,
                  readout::Symbol=:protein,
                  rng::AbstractRNG=Random.default_rng(),
                  population::Union{Nothing, PopulationConfig}=nothing)

    if population !== nothing
        error("SSA with population dynamics not yet implemented. Use BinomialTauLeap or CLE.")
    end

    G = net.G
    bursty = is_bursty(net)

    # With telegraph: 4G chemical + 2G switching = 6G reactions
    # Without: 4G reactions
    n_reactions = bursty ? 6 * G : 4 * G

    out_cols = readout == :both ? 2G : G
    Y = Matrix{Float64}(undef, cell_num, out_cols)

    m_ss, p_ss = steady_state(net, kin)

    for cell in 1:cell_num
        m = round.(Int, m_ss)
        p = round.(Int, p_ss)

        # Promoter state: true = ON
        promoter_on = trues(G)
        if bursty
            for i in 1:G
                if isfinite(net.k_on[i]) && isfinite(net.k_off[i])
                    promoter_on[i] = rand(rng) < net.k_on[i] / (net.k_on[i] + net.k_off[i])
                end
            end
        end

        t = 0.0
        propensities = Vector{Float64}(undef, n_reactions)

        while t < T
            a_total = 0.0
            for i in 1:G
                reg = regulatory_input(net, Float64.(p), i, kin.K_d, kin.n)

                # Reaction 1: Transcription (gated by promoter state)
                prop_transcription = promoter_on[i] ? max(reg + net.basals[i], 0.0) : 0.0
                propensities[4*(i-1) + 1] = prop_transcription
                a_total += prop_transcription

                # Reaction 2: mRNA decay
                prop_mrna_decay = (kin.mu_m + kin.dilution) * m[i]
                propensities[4*(i-1) + 2] = prop_mrna_decay
                a_total += prop_mrna_decay

                # Reaction 3: Translation
                prop_translation = kin.k_t * m[i]
                propensities[4*(i-1) + 3] = prop_translation
                a_total += prop_translation

                # Reaction 4: Protein decay
                prop_protein_decay = (kin.mu_p + kin.dilution) * p[i]
                propensities[4*(i-1) + 4] = prop_protein_decay
                a_total += prop_protein_decay
            end

            # Telegraph switching reactions (if bursty)
            if bursty
                for i in 1:G
                    # Reaction 4G + 2(i-1) + 1: OFF → ON (rate k_on)
                    prop_on = (!promoter_on[i] && isfinite(net.k_on[i])) ? net.k_on[i] : 0.0
                    propensities[4*G + 2*(i-1) + 1] = prop_on
                    a_total += prop_on

                    # Reaction 4G + 2(i-1) + 2: ON → OFF (rate k_off)
                    prop_off = (promoter_on[i] && isfinite(net.k_off[i])) ? net.k_off[i] : 0.0
                    propensities[4*G + 2*(i-1) + 2] = prop_off
                    a_total += prop_off
                end
            end

            if a_total <= 0.0
                break
            end

            tau = -log(rand(rng)) / a_total
            t += tau
            if t > T
                break
            end

            # Choose reaction
            r = rand(rng) * a_total
            cumsum_val = 0.0
            reaction_idx = 0
            for k in 1:n_reactions
                cumsum_val += propensities[k]
                if cumsum_val >= r
                    reaction_idx = k
                    break
                end
            end
            if reaction_idx == 0
                reaction_idx = n_reactions
            end

            # Apply reaction
            if reaction_idx <= 4 * G
                gene = div(reaction_idx - 1, 4) + 1
                rxn_type = mod1(reaction_idx, 4)

                if rxn_type == 1
                    m[gene] += 1
                elseif rxn_type == 2
                    m[gene] = max(m[gene] - 1, 0)
                elseif rxn_type == 3
                    p[gene] += 1
                else
                    p[gene] = max(p[gene] - 1, 0)
                end
            else
                # Telegraph switching reaction
                switch_idx = reaction_idx - 4 * G
                gene = div(switch_idx - 1, 2) + 1
                switch_type = mod1(switch_idx, 2)

                if switch_type == 1
                    promoter_on[gene] = true   # OFF → ON
                else
                    promoter_on[gene] = false  # ON → OFF
                end
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

# ── Single-gene birth-death convenience ──────────────────────────

"""
    simulate_birth_death(birth_rate, death_rate, T; n_samples, x0, rng)

Exact SSA for a simple birth-death process:
  ∅ → X   rate = birth_rate
  X → ∅   rate = death_rate * X

Analytical steady state: mean = birth_rate/death_rate,
variance = birth_rate/death_rate (Poisson).

Returns a vector of n_samples final counts.
"""
function simulate_birth_death(birth_rate::Float64, death_rate::Float64, T::Float64;
                              n_samples::Int=10000,
                              x0::Int=0,
                              rng::AbstractRNG=Random.default_rng())
    results = Vector{Int}(undef, n_samples)

    for s in 1:n_samples
        x = x0 == 0 ? round(Int, birth_rate / death_rate) : x0
        t = 0.0

        while t < T
            a_birth = birth_rate
            a_death = Float64(death_rate) * x
            a_total = a_birth + a_death

            if a_total <= 0.0
                break
            end

            tau = -log(rand(rng)) / a_total
            t += tau
            if t > T
                break
            end

            if rand(rng) * a_total < a_birth
                x += 1
            else
                x = max(x - 1, 0)
            end
        end

        results[s] = x
    end

    return results
end
