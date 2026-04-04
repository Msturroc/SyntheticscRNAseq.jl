#= ================================================================
   Gene regulatory network representation.

   GeneNetwork stores the topology and regulatory strengths of a
   G-gene network. Hill-function regulation with activation,
   repression, cooperative (AND-gate), and redundant (OR-gate) edges.
   ================================================================ =#

"""
    CoopEdge(target, sources, strength)

Cooperative (AND-gate) regulation: gene `target` is activated only
when ALL source proteins are bound. Effective regulation is
`strength * prod(hill_activation(p[src], K, n) for src in sources)`.
"""
struct CoopEdge
    target::Int
    sources::Vector{Int}
    strength::Float64
end

"""
    RedunEdge(target, sources, strength)

Redundant (OR-gate) regulation: gene `target` is activated when
ANY source protein is bound. Uses inclusion-exclusion:
`strength * (1 - prod(1 - hill_activation(p[src], K, n) for src in sources))`.
For two sources this reduces to `strength * (h1 + h2 - h1*h2)`.
"""
struct RedunEdge
    target::Int
    sources::Vector{Int}
    strength::Float64
end

"""
    GeneNetwork(G, basals, interactions[, cooperative, redundant, k_on, k_off])

A gene regulatory network with G genes.

Fields:
- `G`: number of genes
- `basals`: basal transcription rates β[1:G]
- `interactions`: interaction matrix A[i,j] = effect of protein j on gene i
  (>0 activation, <0 repression, 0 no regulation). Diagonal is zero.
- `cooperative`: AND-gate pairs (optional)
- `redundant`: OR-gate pairs (optional)
- `k_on`: promoter activation rates (telegraph model). Inf = always ON (constitutive).
- `k_off`: promoter deactivation rates (telegraph model). Inf = always ON (constitutive).

Telegraph model (Peccoud & Ycart 1995): each gene switches between ON (transcribes
at rate β) and OFF (no transcription). Burst size b = β/k_off, burst frequency f = k_on.
mRNA Fano factor F = 1 + b/(1 + f/k_off + μ_m/k_off).
"""
struct GeneNetwork
    G::Int
    basals::Vector{Float64}
    interactions::Matrix{Float64}
    cooperative::Vector{CoopEdge}
    redundant::Vector{RedunEdge}
    k_on::Vector{Float64}
    k_off::Vector{Float64}
end

function GeneNetwork(G::Int, basals::Vector{Float64}, interactions::Matrix{Float64})
    @assert length(basals) == G
    @assert size(interactions) == (G, G)
    @assert all(interactions[i, i] == 0.0 for i in 1:G)
    GeneNetwork(G, basals, interactions, CoopEdge[], RedunEdge[],
                fill(Inf, G), fill(Inf, G))
end

function GeneNetwork(G::Int, basals::Vector{Float64}, interactions::Matrix{Float64},
                     cooperative::Vector{CoopEdge}, redundant::Vector{RedunEdge})
    GeneNetwork(G, basals, interactions, cooperative, redundant,
                fill(Inf, G), fill(Inf, G))
end

"""
    GeneNetwork(basals, interactions[; cooperative, redundant, k_on, k_off])

Convenience constructor that infers G from basals.
Optionally provide cooperative/redundant edges and telegraph switching rates.
"""
function GeneNetwork(basals::Vector{Float64}, interactions::Matrix{Float64};
                     cooperative::Vector{CoopEdge}=CoopEdge[],
                     redundant::Vector{RedunEdge}=RedunEdge[],
                     k_on::Union{Nothing, Vector{Float64}}=nothing,
                     k_off::Union{Nothing, Vector{Float64}}=nothing)
    G = length(basals)
    kon = k_on === nothing ? fill(Inf, G) : k_on
    koff = k_off === nothing ? fill(Inf, G) : k_off
    @assert length(kon) == G
    @assert length(koff) == G
    GeneNetwork(G, basals, interactions, cooperative, redundant, kon, koff)
end

"""
    is_bursty(network)

Returns true if any gene has finite (non-Inf) telegraph switching rates.
"""
is_bursty(net::GeneNetwork) = any(isfinite, net.k_on) || any(isfinite, net.k_off)

# ── Hill function ────────────────────────────────────────────────

"""
    hill_activation(p, K, n)

Hill activation function: p^n / (p^n + K^n).
"""
@inline function hill_activation(p::Float64, K::Float64, n::Float64)::Float64
    pn = p^n
    return pn / (pn + K^n)
end

"""
    hill_repression(p, K, n)

Hill repression function: K^n / (p^n + K^n).
"""
@inline function hill_repression(p::Float64, K::Float64, n::Float64)::Float64
    pn = p^n
    return K^n / (pn + K^n)
end

"""
    hill_regulate(p, a, K, n)

Compute Hill regulation for interaction strength `a`:
- a > 0: activation with strength a
- a < 0: repression with strength |a|
- a == 0: no regulation (returns 0)
"""
@inline function hill_regulate(p::Float64, a::Float64, K::Float64, n::Float64)::Float64
    if a > 0.0
        return a * hill_activation(p, K, n)
    elseif a < 0.0
        return (-a) * hill_repression(p, K, n)
    else
        return 0.0
    end
end

# ── Regulatory input computation ─────────────────────────────────

"""
    regulatory_input(network, proteins, gene_i, K, n)

Compute the total regulatory input for gene `gene_i` from all
other genes' protein concentrations, using Hill functions.

Combines three regulation types:
- **Additive**: independent Hill contributions summed
- **Cooperative (AND)**: `strength * prod(hill(p[src]))` — all sources must be bound
- **Redundant (OR)**: `strength * (1 - prod(1 - hill(p[src])))` — any source suffices
"""
function regulatory_input(net::GeneNetwork, proteins::Vector{Float64},
                          gene_i::Int, K::Float64, n::Float64)::Float64
    reg = 0.0

    # Additive Hill regulation
    for j in 1:net.G
        if j != gene_i
            a_ij = net.interactions[gene_i, j]
            if a_ij != 0.0
                reg += hill_regulate(proteins[j], a_ij * net.basals[gene_i], K, n)
            end
        end
    end

    # Cooperative (AND-gate): strength * β[i] * prod(hill_activation)
    for edge in net.cooperative
        edge.target == gene_i || continue
        prod_h = 1.0
        for src in edge.sources
            prod_h *= hill_activation(proteins[src], K, n)
        end
        reg += edge.strength * net.basals[gene_i] * prod_h
    end

    # Redundant (OR-gate): strength * β[i] * (1 - prod(1 - hill_activation))
    for edge in net.redundant
        edge.target == gene_i || continue
        prod_1mh = 1.0
        for src in edge.sources
            prod_1mh *= (1.0 - hill_activation(proteins[src], K, n))
        end
        reg += edge.strength * net.basals[gene_i] * (1.0 - prod_1mh)
    end

    return reg
end

# ── Vectorized Hill (for CPU CLE / tau-leap) ─────────────────────

"""
    precompute_hill_matrices(network)

Decompose the effective interaction matrix A_eff[i,j] = A[i,j] * β[i]
into positive (activation) and negative (repression) parts.

Returns (A_pos, A_neg) for vectorized Hill computation:
  reg_input = A_pos * act_frac + A_neg * rep_frac
"""
function precompute_hill_matrices(net::GeneNetwork)
    G = net.G
    A_eff = zeros(G, G)
    for i in 1:G
        for j in 1:G
            if i != j
                A_eff[i, j] = net.interactions[i, j] * net.basals[i]
            end
        end
    end
    A_pos = max.(A_eff, 0.0)
    A_neg = max.(-A_eff, 0.0)
    return A_pos, A_neg
end

# ── Construction from parameter vector θ ─────────────────────────

"""
    GeneNetwork(θ::Vector{Float64}, G::Int)

Construct a GeneNetwork from a flat parameter vector θ.

If length(θ) == G*(G-1) + G: first G entries are log-basals, rest are interactions.
If length(θ) == G*(G-1): interactions only, basals default to exp(-1).

Interaction ordering: row-major over off-diagonal entries of A[i,j].
"""
function GeneNetwork(θ::Vector{Float64}, G::Int)
    n_interactions = G * (G - 1)
    if length(θ) == n_interactions + G
        basals = exp.(θ[1:G])
        a_vec = θ[G+1:end]
    elseif length(θ) == n_interactions
        basals = fill(exp(-1.0), G)
        a_vec = θ
    else
        error("θ must have length $n_interactions or $(n_interactions + G), got $(length(θ))")
    end

    A = zeros(G, G)
    idx = 1
    for i in 1:G
        for j in 1:G
            if i != j
                A[i, j] = a_vec[idx]
                idx += 1
            end
        end
    end
    return GeneNetwork(G, basals, A)
end

"""
    n_species(network)

Total number of molecular species: G mRNAs + G proteins = 2G.
"""
n_species(net::GeneNetwork) = 2 * net.G
