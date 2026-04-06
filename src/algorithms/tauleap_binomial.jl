#= ================================================================
   Binomial tau-leap — naturally non-negative.

   For each reaction with propensity a and available molecules n:
     events ~ Binomial(n, 1 - exp(-a/n * dt))   for decay/conversion
     events ~ Poisson(a * dt)                    for creation (unlimited supply)

   Key advantage over Poisson tau-leap: decay events cannot exceed
   available molecules, so counts never go negative. No clamping bias.

   Reference: Chatterjee, Vlachos, Katsoulakis (2005) "Binomial
   distribution based tau-leap accelerated stochastic simulation"
   ================================================================ =#

"""
    BinomialTauLeap(dt)

Binomial tau-leap algorithm. Naturally non-negative.
"""
struct BinomialTauLeap
    dt::Float64
end

# ── Single-cell (no population) ──────────────────────────────────

"""
    simulate(network, alg::BinomialTauLeap, kinetics; cell_num, T, readout, rng, population)

Binomial tau-leap simulation. When `population` is provided, simulates
with Moran dynamics and volume-dependent transcription.
"""
function simulate(net::GeneNetwork, alg::BinomialTauLeap, kin::KineticParams;
                  cell_num::Int=1000,
                  T::Float64=50.0,
                  readout::Symbol=:protein,
                  rng::AbstractRNG=Random.default_rng(),
                  population::Union{Nothing, PopulationConfig}=nothing,
                  regulation_mode::Symbol=:transcription)

    if population !== nothing
        return _simulate_population(net, alg, kin, population;
                                    T=T, readout=readout, rng=rng)
    end

    @assert regulation_mode in (:transcription, :switching) "regulation_mode must be :transcription or :switching"
    if regulation_mode == :switching
        @assert is_bursty(net) "switching regulation mode requires finite k_on/k_off"
    end

    G = net.G
    dt = alg.dt
    n_steps = ceil(Int, T / dt)
    out_cols = readout == :both ? 2G : G

    m_ss, p_ss = steady_state(net, kin)
    Y = Matrix{Float64}(undef, cell_num, out_cols)

    bursty = is_bursty(net)

    for cell in 1:cell_num
        m = round.(Int, m_ss)
        p = round.(Int, p_ss)

        # Telegraph promoter state: true = ON, false = OFF
        # Initialize in steady-state: P(ON) = k_on / (k_on + k_off)
        if bursty
            promoter_on = BitVector(undef, G)
            for i in 1:G
                if isfinite(net.k_on[i]) && isfinite(net.k_off[i])
                    p_on = net.k_on[i] / (net.k_on[i] + net.k_off[i])
                    promoter_on[i] = rand(rng) < p_on
                else
                    promoter_on[i] = true  # constitutive = always ON
                end
            end
        end

        pf = Vector{Float64}(undef, G)

        for step in 1:n_steps
            for g in 1:G; pf[g] = Float64(p[g]); end

            # Telegraph switching (if any gene is bursty)
            if bursty
                for i in 1:G
                    isfinite(net.k_on[i]) || continue
                    if regulation_mode == :switching
                        # Protein-dependent switching rates
                        sigma_b = net.k_on[i]
                        sigma_u = net.k_off[i]
                        for j in 1:G
                            j == i && continue
                            aij = net.interactions[i, j]
                            aij == 0.0 && continue
                            if aij > 0.0
                                sigma_b += aij * net.basals[i] * hill_activation(pf[j], kin.K_d, kin.n)
                            else
                                sigma_u += (-aij) * net.basals[i] * hill_activation(pf[j], kin.K_d, kin.n)
                            end
                        end
                        if promoter_on[i]
                            if rand(rng) < (1.0 - exp(-max(sigma_u, 0.0) * dt))
                                promoter_on[i] = false
                            end
                        else
                            if rand(rng) < (1.0 - exp(-max(sigma_b, 0.0) * dt))
                                promoter_on[i] = true
                            end
                        end
                    else
                        if promoter_on[i]
                            # ON → OFF with probability 1 - exp(-k_off * dt)
                            if rand(rng) < (1.0 - exp(-net.k_off[i] * dt))
                                promoter_on[i] = false
                            end
                        else
                            # OFF → ON with probability 1 - exp(-k_on * dt)
                            if rand(rng) < (1.0 - exp(-net.k_on[i] * dt))
                                promoter_on[i] = true
                            end
                        end
                    end
                end
            end

            for i in 1:G
                if regulation_mode == :transcription
                    reg = regulatory_input(net, pf, i, kin.K_d, kin.n)
                    prop_transcription = max(reg + net.basals[i], 0.0)
                else  # :switching — constant rate when ON
                    prop_transcription = net.basals[i]
                end

                # Telegraph gating: transcription only when promoter is ON
                if bursty && !promoter_on[i]
                    prop_transcription = 0.0
                end

                # Transcription: creation from unlimited supply → Poisson
                n_transcribe = rand(rng, Distributions.Poisson(prop_transcription * dt))

                # mRNA decay (intrinsic + dilution): bounded by available mRNA → Binomial
                if m[i] > 0
                    p_decay_m = 1.0 - exp(-(kin.mu_m + kin.dilution) * dt)
                    n_mrna_decay = rand(rng, Distributions.Binomial(m[i], p_decay_m))
                else
                    n_mrna_decay = 0
                end

                m[i] += n_transcribe - n_mrna_decay

                # Translation: creation from unlimited supply → Poisson
                n_translate = rand(rng, Distributions.Poisson(kin.k_t * max(m[i], 0) * dt))

                # Protein decay (intrinsic + dilution): bounded by available protein → Binomial
                if p[i] > 0
                    p_decay_p = 1.0 - exp(-(kin.mu_p + kin.dilution) * dt)
                    n_prot_decay = rand(rng, Distributions.Binomial(p[i], p_decay_p))
                else
                    n_prot_decay = 0
                end

                p[i] += n_translate - n_prot_decay
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

# ── Population simulation (Moran + volume-dependent transcription) ──

"""
    _simulate_population(net, alg::BinomialTauLeap, kin, pop; T, readout, rng)

Binomial tau-leap with Moran population dynamics.

Key differences from single-cell:
- Transcription rate scales with cell volume: prop = (reg + β) * V
- No explicit dilution term (dilution is emergent from division)
- mRNA/protein decay use intrinsic rates only (mu_m, mu_p)
- Exponential volume growth + binomial division + Moran replacement
"""
function _simulate_population(net::GeneNetwork, alg::BinomialTauLeap,
                              kin::KineticParams, pop::PopulationConfig;
                              T::Float64=50.0,
                              readout::Symbol=:protein,
                              rng::AbstractRNG=Random.default_rng())
    G = net.G
    N = pop.cell_num
    dt = alg.dt
    n_steps = ceil(Int, T / dt)

    state = initialize_population(net, kin, pop; rng=rng)

    p_float = Vector{Float64}(undef, G)  # reusable buffer for Hill computation

    for step in 1:n_steps
        # ── Chemical kinetics for all cells ──
        for c in 1:N
            V = state.volumes[c]

            # Copy protein to float for Hill function
            for g in 1:G
                p_float[g] = Float64(state.protein[g, c])
            end

            for i in 1:G
                reg = regulatory_input(net, p_float, i, kin.K_d, kin.n)

                # Volume-dependent transcription: prop = (reg + β) * V
                prop_transcription = max(reg + net.basals[i], 0.0) * V

                # Transcription → Poisson (creation from unlimited supply)
                n_transcribe = rand(rng, Distributions.Poisson(prop_transcription * dt))

                # mRNA decay: intrinsic rate only (no dilution term)
                if state.mrna[i, c] > 0
                    p_decay_m = 1.0 - exp(-kin.mu_m * dt)
                    n_mrna_decay = rand(rng, Distributions.Binomial(state.mrna[i, c], p_decay_m))
                else
                    n_mrna_decay = 0
                end

                state.mrna[i, c] += n_transcribe - n_mrna_decay

                # Translation → Poisson
                n_translate = rand(rng, Distributions.Poisson(
                    kin.k_t * max(state.mrna[i, c], 0) * dt))

                # Protein decay: intrinsic rate only (no dilution term)
                if state.protein[i, c] > 0
                    p_decay_p = 1.0 - exp(-kin.mu_p * dt)
                    n_prot_decay = rand(rng, Distributions.Binomial(state.protein[i, c], p_decay_p))
                else
                    n_prot_decay = 0
                end

                state.protein[i, c] += n_translate - n_prot_decay
            end
        end

        # ── Volume growth ──
        grow_volumes!(state, pop, dt; rng=rng)

        # ── Division check (every div_check_interval steps) ──
        if step % pop.div_check_interval == 0
            division_check!(state, pop; rng=rng)
        end
    end

    return extract_snapshot(state, readout)
end

"""
    simulate_with_state(net, alg::BinomialTauLeap, kin, pop; T, readout, rng)

Like `simulate` with population, but returns `(Y, state::PopulationState)`
so callers can access final volumes for validation.
"""
function simulate_with_state(net::GeneNetwork, alg::BinomialTauLeap,
                             kin::KineticParams, pop::PopulationConfig;
                             T::Float64=50.0,
                             readout::Symbol=:protein,
                             rng::AbstractRNG=Random.default_rng())
    G = net.G
    N = pop.cell_num
    dt = alg.dt
    n_steps = ceil(Int, T / dt)

    state = initialize_population(net, kin, pop; rng=rng)

    p_float = Vector{Float64}(undef, G)

    for step in 1:n_steps
        for c in 1:N
            V = state.volumes[c]
            for g in 1:G
                p_float[g] = Float64(state.protein[g, c])
            end

            for i in 1:G
                reg = regulatory_input(net, p_float, i, kin.K_d, kin.n)
                prop_transcription = max(reg + net.basals[i], 0.0) * V
                n_transcribe = rand(rng, Distributions.Poisson(prop_transcription * dt))

                if state.mrna[i, c] > 0
                    p_decay_m = 1.0 - exp(-kin.mu_m * dt)
                    n_mrna_decay = rand(rng, Distributions.Binomial(state.mrna[i, c], p_decay_m))
                else
                    n_mrna_decay = 0
                end
                state.mrna[i, c] += n_transcribe - n_mrna_decay

                n_translate = rand(rng, Distributions.Poisson(
                    kin.k_t * max(state.mrna[i, c], 0) * dt))

                if state.protein[i, c] > 0
                    p_decay_p = 1.0 - exp(-kin.mu_p * dt)
                    n_prot_decay = rand(rng, Distributions.Binomial(state.protein[i, c], p_decay_p))
                else
                    n_prot_decay = 0
                end
                state.protein[i, c] += n_translate - n_prot_decay
            end
        end

        grow_volumes!(state, pop, dt; rng=rng)
        if step % pop.div_check_interval == 0
            division_check!(state, pop; rng=rng)
        end
    end

    return extract_snapshot(state, readout), state
end
