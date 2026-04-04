#= ================================================================
   Kinetic parameters for gene expression dynamics.

   Default values from Jorgensen et al. (2013) / SLInG / Blair (2025).
   ================================================================ =#

"""
    KineticParams

Fixed kinetic parameters shared across all genes (for now).

Fields:
- `k_t`: translation rate (mRNA -> protein)
- `K_d`: Hill dissociation constant
- `n`: Hill coefficient
- `mu_m`: mRNA degradation rate
- `mu_p`: protein degradation rate
- `dilution`: growth-coupled dilution rate (default 0). When >0, adds
  a dilution term to both mRNA and protein decay, modelling the effect
  of exponential cell growth without explicit population dynamics.
  Analytical steady state with dilution μ:
    <m> = β/(μ_m + μ), <p> = k_t·<m>/(μ_p + μ),
    F(p) = 1 + k_t/(μ_m + μ_p + 2μ)
"""
struct KineticParams
    k_t::Float64    # translation rate
    K_d::Float64    # Hill dissociation constant
    n::Float64      # Hill coefficient
    mu_m::Float64   # mRNA decay rate
    mu_p::Float64   # protein decay rate
    dilution::Float64  # growth-coupled dilution rate
end

"""
    KineticParams(; k_t, K_d, n, mu_m, mu_p, dilution=0.0)

Construct KineticParams with default values from Jorgensen/SLInG.
Set `dilution > 0` to model growth-coupled dilution without explicit
population dynamics (Sturrock & Sturrock 2026).
"""
function KineticParams(;
    k_t::Float64  = 4.3376521035785345,
    K_d::Float64  = 61.47011276535837,
    n::Float64    = 10.0,
    mu_m::Float64 = 0.04987442692380366,
    mu_p::Float64 = 0.36777724535966944,
    dilution::Float64 = 0.0,
)
    KineticParams(k_t, K_d, n, mu_m, mu_p, dilution)
end

# ── Steady-state computation ─────────────────────────────────────

"""
    steady_state(network, kinetics)

Compute the deterministic unregulated steady-state:
  mRNA_ss[i]   = β[i] / (μ_m + μ)
  protein_ss[i] = k_t * mRNA_ss[i] / (μ_p + μ)

where μ = kinetics.dilution (0 by default).

Returns (m_ss, p_ss) vectors of length G.
"""
function steady_state(net::GeneNetwork, kin::KineticParams)
    mu = kin.dilution
    m_ss = net.basals ./ (kin.mu_m + mu)
    p_ss = kin.k_t .* m_ss ./ (kin.mu_p + mu)
    return m_ss, p_ss
end

# ── Prior sampling ───────────────────────────────────────────────

"""
    sample_network(G; rng, regulation, n_tf_range, targets_per_tf_range,
                   n_coop_range, n_redun_range, infer_basals)

Sample a random sparse gene regulatory network with TF-level sparsity.

`regulation` controls which edge types are sampled:
- `:additive` — standard independent Hill edges only
- `:cooperative` — additive + AND-gate pairs
- `:redundant` — additive + OR-gate pairs
- `:mixed` (default) — additive + cooperative + redundant

Returns a GeneNetwork.
"""
function sample_network(G::Int;
                        rng::AbstractRNG=Random.default_rng(),
                        regulation::Symbol=:mixed,
                        n_tf_range::UnitRange{Int} = G <= 5 ? (2:3) : (4:5),
                        targets_per_tf_range::UnitRange{Int} = 3:6,
                        n_coop_range::UnitRange{Int} = 0:1,
                        n_redun_range::UnitRange{Int} = 0:1,
                        infer_basals::Bool=true)
    n_interactions = G * (G - 1)

    # Sample basals
    if infer_basals
        basals = exp.(randn(rng, G) .* 0.5 .- 1.0)  # LogNormal centered near exp(-1)
    else
        basals = fill(exp(-1.0), G)
    end

    # Sample sparse interaction matrix via TF-level sparsity
    A = zeros(G, G)
    n_tf = rand(rng, n_tf_range)
    active_tfs = sort(StatsBase.sample(rng, 1:G, n_tf, replace=false))
    used_pairs = Set{Tuple{Int,Int}}()  # track (target, tf) to avoid overlap

    for tf in active_tfs
        n_targets = rand(rng, targets_per_tf_range)
        possible_targets = [g for g in 1:G if g != tf]
        n_targets = min(n_targets, length(possible_targets))
        targets = StatsBase.sample(rng, possible_targets, n_targets, replace=false)

        for target in targets
            # Edge strength from U([-10,-3] ∪ [3,10])
            if rand(rng) < 0.5
                A[target, tf] = rand(rng, Distributions.Uniform(-10.0, -3.0))
            else
                A[target, tf] = rand(rng, Distributions.Uniform(3.0, 10.0))
            end
            push!(used_pairs, (target, tf))
        end
    end

    # Cooperative and redundant edges
    coop_edges = CoopEdge[]
    redun_edges = RedunEdge[]

    if regulation in (:mixed, :cooperative)
        n_coop = rand(rng, n_coop_range)
        for _ in 1:n_coop
            result = _sample_pair_edge(G, used_pairs, rng)
            if result !== nothing
                target, sources, strength = result
                push!(coop_edges, CoopEdge(target, sources, strength))
            end
        end
    end

    if regulation in (:mixed, :redundant)
        n_redun = rand(rng, n_redun_range)
        for _ in 1:n_redun
            result = _sample_pair_edge(G, used_pairs, rng)
            if result !== nothing
                target, sources, strength = result
                push!(redun_edges, RedunEdge(target, sources, strength))
            end
        end
    end

    # For mixed regulation, guarantee at least one non-additive edge
    if regulation == :mixed && isempty(coop_edges) && isempty(redun_edges)
        result = _sample_pair_edge(G, used_pairs, rng)
        if result !== nothing
            target, sources, strength = result
            push!(redun_edges, RedunEdge(target, sources, strength))
        end
    end

    return GeneNetwork(G, basals, A, coop_edges, redun_edges,
                       fill(Inf, G), fill(Inf, G))
end

"""Sample a cooperative/redundant edge: pick 2 TFs and a target not already used."""
function _sample_pair_edge(G::Int, used_pairs::Set{Tuple{Int,Int}},
                           rng::AbstractRNG)
    G < 3 && return nothing
    for _ in 1:20  # max attempts
        sources = sort(StatsBase.sample(rng, 1:G, 2, replace=false))
        possible_targets = [g for g in 1:G if g != sources[1] && g != sources[2]]
        isempty(possible_targets) && continue
        target = rand(rng, possible_targets)
        pair1 = (target, sources[1])
        pair2 = (target, sources[2])
        if pair1 ∉ used_pairs && pair2 ∉ used_pairs
            push!(used_pairs, pair1)
            push!(used_pairs, pair2)
            strength = rand(rng) < 0.5 ?
                rand(rng, Distributions.Uniform(-10.0, -3.0)) :
                rand(rng, Distributions.Uniform(3.0, 10.0))
            return (target, sources, strength)
        end
    end
    return nothing
end
