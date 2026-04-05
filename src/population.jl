#= ================================================================
   Cell population dynamics: Moran-like fixed-population process.

   Exponential volume growth, volume-dependent transcription,
   binomial molecule partitioning at division, Moran replacement
   (daughter replaces a uniformly random cell).

   Key design: when population dynamics are active, there is NO
   explicit dilution term in the chemical kinetics. Dilution is
   emergent from binomial partitioning at division. This avoids
   double-counting and matches the discrete Moran process exactly
   (Sturrock & Sturrock 2026, Appendix E).
   ================================================================ =#

"""
    PopulationConfig

Configuration for Moran-like fixed-population dynamics.

Fields:
- `cell_num`: fixed population size (N)
- `growth_rate`: exponential volume growth rate λ
- `V_div`: division volume threshold
- `V_init`: initial volume range (uniform)
- `div_check_interval`: simulation steps between division checks
"""
struct PopulationConfig
    cell_num::Int
    growth_rate::Float64
    V_div::Float64
    V_init::Tuple{Float64, Float64}
    div_check_interval::Int
    growth_noise::Float64
end

function PopulationConfig(;
    cell_num::Int = 1000,
    growth_rate::Float64 = 0.03,
    V_div::Float64 = 2.0,
    V_init::Tuple{Float64, Float64} = (0.8, 1.2),
    div_check_interval::Int = 10,
    growth_noise::Float64 = 0.0,
)
    PopulationConfig(cell_num, growth_rate, V_div, V_init, div_check_interval, growth_noise)
end

"""
    CaptureModel

scRNA-seq capture/dropout model (LogNormal-Binomial).

Fields:
- `efficiency`: mean capture rate β
- `efficiency_std`: LogNormal cell-to-cell variation in capture
- `readout`: which species to capture (:mrna or :protein)
"""
struct CaptureModel
    efficiency::Float64
    efficiency_std::Float64
    readout::Symbol

    function CaptureModel(efficiency::Float64, efficiency_std::Float64, readout::Symbol)
        @assert readout in (:mrna, :protein) "readout must be :mrna or :protein"
        new(efficiency, efficiency_std, readout)
    end
end

function CaptureModel(;
    efficiency::Float64 = 0.1,
    efficiency_std::Float64 = 0.2,
    readout::Symbol = :mrna,
)
    CaptureModel(efficiency, efficiency_std, readout)
end

# ── Population state ─────────────────────────────────────────────

"""
    PopulationState

Mutable state for a cell population during simulation.
All arrays are (G, N) for mRNA/protein, (N,) for volumes.
"""
mutable struct PopulationState
    mrna::Matrix{Int}       # (G, N) mRNA counts
    protein::Matrix{Int}    # (G, N) protein counts
    volumes::Vector{Float64} # (N,) cell volumes
end

"""
    initialize_population(network, kinetics, pop; rng)

Create initial population state: steady-state molecule counts,
random initial volumes from U(V_init[1], V_init[2]).
"""
function initialize_population(net::GeneNetwork, kin::KineticParams,
                               pop::PopulationConfig;
                               rng::AbstractRNG=Random.default_rng())
    G = net.G
    N = pop.cell_num
    m_ss, p_ss = steady_state(net, kin)

    mrna = Matrix{Int}(undef, G, N)
    protein = Matrix{Int}(undef, G, N)
    for c in 1:N
        for g in 1:G
            mrna[g, c] = round(Int, m_ss[g])
            protein[g, c] = round(Int, p_ss[g])
        end
    end

    V_lo, V_hi = pop.V_init
    volumes = V_lo .+ (V_hi - V_lo) .* rand(rng, N)

    return PopulationState(mrna, protein, volumes)
end

# ── Division + Moran replacement ─────────────────────────────────

"""
    grow_volumes!(state, pop, dt; rng)

Exponential volume growth. If `growth_noise > 0`, each cell gets an
independent Gaussian perturbation to its growth rate, breaking the
deterministic synchronization that otherwise creates persistent
cell-cycle waves in the Moran process.
"""
function grow_volumes!(state::PopulationState, pop::PopulationConfig, dt::Float64;
                       rng::AbstractRNG=Random.default_rng())
    if pop.growth_noise > 0.0
        σ = pop.growth_noise
        for c in eachindex(state.volumes)
            state.volumes[c] *= exp((pop.growth_rate + σ * randn(rng)) * dt)
        end
    else
        factor = exp(pop.growth_rate * dt)
        state.volumes .*= factor
    end
end

"""
    division_check!(state, pop; rng)

Check all cells for division (V > V_div) and perform Moran replacement.

Two-phase process:
  Phase 1: Identify dividers, do binomial partitioning, store daughters
  Phase 2: Each daughter replaces a uniformly random cell (including
           possibly its own mother — uniform death is age-independent)

Note: with deterministic growth, finite-N Moran always exhibits
persistent volume fluctuations (critical branching number = 1).
The analytical p(v) = V_div/v² is an ensemble average.

Returns the number of divisions that occurred.
"""
function division_check!(state::PopulationState, pop::PopulationConfig;
                         rng::AbstractRNG=Random.default_rng())
    G = size(state.mrna, 1)
    N = pop.cell_num

    dividers = findall(v -> v > pop.V_div, state.volumes)
    n_div = length(dividers)
    n_div == 0 && return 0

    # Phase 1: partition molecules and halve mother volumes
    daughter_mrna = Matrix{Int}(undef, G, n_div)
    daughter_protein = Matrix{Int}(undef, G, n_div)
    daughter_volumes = Vector{Float64}(undef, n_div)

    for (k, cell) in enumerate(dividers)
        for g in 1:G
            if state.mrna[g, cell] > 0
                m_d = rand(rng, Distributions.Binomial(state.mrna[g, cell], 0.5))
                state.mrna[g, cell] -= m_d
                daughter_mrna[g, k] = m_d
            else
                daughter_mrna[g, k] = 0
            end

            if state.protein[g, cell] > 0
                p_d = rand(rng, Distributions.Binomial(state.protein[g, cell], 0.5))
                state.protein[g, cell] -= p_d
                daughter_protein[g, k] = p_d
            else
                daughter_protein[g, k] = 0
            end
        end

        state.volumes[cell] /= 2.0
        daughter_volumes[k] = state.volumes[cell]
    end

    # Phase 2: each daughter replaces a uniformly random cell
    for k in 1:n_div
        replaced = rand(rng, 1:N)
        for g in 1:G
            state.mrna[g, replaced] = daughter_mrna[g, k]
            state.protein[g, replaced] = daughter_protein[g, k]
        end
        state.volumes[replaced] = daughter_volumes[k]
    end

    return n_div
end

# ── Snapshot extraction ──────────────────────────────────────────

"""
    extract_snapshot(state, readout)

Extract (cell_num x G) or (cell_num x 2G) matrix from population state.
"""
function extract_snapshot(state::PopulationState, readout::Symbol)
    G, N = size(state.mrna)
    if readout == :mrna
        return Float64.(state.mrna')
    elseif readout == :protein
        return Float64.(state.protein')
    else  # :both
        return hcat(Float64.(state.mrna'), Float64.(state.protein'))
    end
end
