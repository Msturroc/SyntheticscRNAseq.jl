#= ================================================================
   Holimap: semi-analytical steady-state marginal distributions
   for multi-gene regulatory networks.

   Based on Cao, Luo & Grima (2024) "Holimap: an accurate and
   efficient method for solving stochastic gene network dynamics"
   Nature Communications.

   Implements 2nd-order Holimap (2-HM):
   1. Fixed-point iteration for effective switching rates
   2. 2nd-order moment correction for Hill function expectations
   3. Per-gene telegraph marginals via existing telegraph_distribution

   Model: switching-rate regulation
   - Transcription rate (ON): ρ_i = basals[i] (constant)
   - OFF→ON rate: σ_b_i = k_on[i] + Σ activation contributions
   - ON→OFF rate: σ_u_i = k_off[i] + Σ repression contributions

   Regulation enters through switching propensities, not transcription.
   For a single unregulated gene, exactly recovers telegraph_distribution.
   ================================================================ =#

"""
    HolimapResult

Result from `holimap_marginals`: per-gene marginal distributions
and moments computed via the Holimap method.

Fields:
- `marginals`: Vector of per-gene PMF vectors P(m=0), P(m=1), ...
- `means`: Per-gene mRNA means
- `variances`: Per-gene mRNA variances
- `protein_means`: Per-gene protein means (two-stage model)
- `protein_variances`: Per-gene protein variances (two-stage model)
- `effective_a`: Effective dimensionless ON rate σ_b/μ_m per gene
- `effective_b`: Effective dimensionless OFF rate σ_u/μ_m per gene
- `effective_c`: Dimensionless transcription rate ρ/μ_m per gene
- `converged`: Whether fixed-point iteration converged
- `iterations`: Number of iterations used
"""
struct HolimapResult
    marginals::Vector{Vector{Float64}}
    means::Vector{Float64}
    variances::Vector{Float64}
    protein_means::Vector{Float64}
    protein_variances::Vector{Float64}
    effective_a::Vector{Float64}
    effective_b::Vector{Float64}
    effective_c::Vector{Float64}
    converged::Bool
    iterations::Int
end

"""
    holimap_marginals(net, kin; nmax=200, maxiter=200, tol=1e-8, damping=0.3)

Compute per-gene marginal mRNA distributions using the 2nd-order
Holimap method (Cao, Luo & Grima 2024).

For a single unregulated gene, exactly recovers `telegraph_distribution`.
For multi-gene networks with switching-rate regulation, computes
self-consistent effective telegraph parameters via fixed-point iteration
with 2nd-order moment closure for Hill function expectations.

The switching-rate model has constant transcription when ON and
protein-dependent promoter switching rates:
- σ_b_i = k_on[i] + Σ_{j: activator} A[i,j] * β[i] * E[h(p_j)]
- σ_u_i = k_off[i] + Σ_{j: repressor} |A[i,j]| * β[i] * E[h(p_j)]

where h is the Hill activation function and the expectation uses a
2nd-order Taylor correction: E[h(p)] ≈ h(μ) + ½h″(μ)σ².

Protein moments are derived from the two-stage model:
  E[p] = k_t E[m] / μ_p,  Var[p] = E[p] + k_t² Var[m] / (μ_p(μ_m+μ_p))

# Arguments
- `net::GeneNetwork`: network with finite k_on, k_off (telegraph model)
- `kin::KineticParams`: kinetic parameters
- `nmax::Int=200`: maximum mRNA count for PMF truncation
- `maxiter::Int=200`: maximum fixed-point iterations
- `tol::Float64=1e-8`: relative convergence tolerance
- `damping::Float64=0.3`: damping factor (0=no update, 1=full update)

# Returns
- `HolimapResult` with per-gene marginals, moments, effective parameters
"""
function holimap_marginals(net::GeneNetwork, kin::KineticParams;
                           nmax::Int=200, maxiter::Int=200,
                           tol::Float64=1e-8, damping::Float64=0.3)
    G = net.G
    d_eff = kin.mu_m + kin.dilution
    mu_p = kin.mu_p + kin.dilution
    k_t = kin.k_t
    K = kin.K_d
    n_hill = kin.n

    # Initialize with unregulated telegraph parameters
    a = Vector{Float64}(undef, G)
    b = Vector{Float64}(undef, G)
    c = Vector{Float64}(undef, G)
    for i in 1:G
        a[i] = isfinite(net.k_on[i]) ? net.k_on[i] / d_eff : 1e6
        b[i] = isfinite(net.k_off[i]) ? net.k_off[i] / d_eff : 1e6
        c[i] = net.basals[i] / d_eff
    end

    # Check if any gene has inter-gene regulation
    has_regulation = false
    for i in 1:G, j in 1:G
        if i != j && net.interactions[i, j] != 0.0
            has_regulation = true
            break
        end
    end

    converged = !has_regulation  # trivially converged if no coupling
    n_iter = 0

    if has_regulation
        for iter in 1:maxiter
            n_iter = iter

            # Compute mRNA moments from current telegraph parameters
            m_means = [telegraph_mean(a[i], b[i], c[i]) for i in 1:G]
            m_vars = [telegraph_variance(a[i], b[i], c[i]) for i in 1:G]

            # Protein moments (two-stage model, linear propagation)
            p_means = k_t .* m_means ./ mu_p
            p_vars = p_means .+ k_t^2 .* m_vars ./ (mu_p * (d_eff + mu_p))

            # Update effective switching rates
            a_new = Vector{Float64}(undef, G)
            b_new = Vector{Float64}(undef, G)

            for i in 1:G
                sigma_b = isfinite(net.k_on[i]) ? net.k_on[i] : 1e6 * d_eff
                sigma_u = isfinite(net.k_off[i]) ? net.k_off[i] : 1e6 * d_eff

                for j in 1:G
                    j == i && continue
                    aij = net.interactions[i, j]
                    aij == 0.0 && continue

                    E_h = _expected_hill(p_means[j], p_vars[j], K, n_hill)

                    if aij > 0.0
                        sigma_b += aij * net.basals[i] * E_h
                    else
                        sigma_u += (-aij) * net.basals[i] * E_h
                    end
                end

                a_new[i] = max(sigma_b, 1e-10) / d_eff
                b_new[i] = max(sigma_u, 1e-10) / d_eff
            end

            # Check convergence
            max_change = 0.0
            for i in 1:G
                max_change = max(max_change, abs(a_new[i] - a[i]) / max(abs(a[i]), 1e-10))
                max_change = max(max_change, abs(b_new[i] - b[i]) / max(abs(b[i]), 1e-10))
            end

            if max_change < tol
                a .= a_new
                b .= b_new
                converged = true
                break
            end

            # Damped update for stability
            for i in 1:G
                a[i] = (1.0 - damping) * a[i] + damping * a_new[i]
                b[i] = (1.0 - damping) * b[i] + damping * b_new[i]
            end
        end
    end

    # Compute final marginals and moments
    marginals = Vector{Vector{Float64}}(undef, G)
    means = Vector{Float64}(undef, G)
    variances = Vector{Float64}(undef, G)

    for i in 1:G
        marginals[i] = telegraph_distribution(a[i], b[i], c[i]; nmax=nmax)
        means[i] = telegraph_mean(a[i], b[i], c[i])
        variances[i] = telegraph_variance(a[i], b[i], c[i])
    end

    # Final protein moments
    p_means = k_t .* means ./ mu_p
    p_vars = p_means .+ k_t^2 .* variances ./ (mu_p * (d_eff + mu_p))

    return HolimapResult(marginals, means, variances, p_means, p_vars,
                         a, b, c, converged, n_iter)
end

# ── Hill function moment correction ─────────────────────────────

"""
    _expected_hill(mu, sigma2, K, n)

Compute E[hill_activation(X, K, n)] where X has mean μ and variance σ².
Uses 2nd-order Taylor expansion:
    E[h(X)] ≈ h(μ) + ½ h″(μ) σ²
"""
function _expected_hill(mu::Float64, sigma2::Float64, K::Float64, n::Float64)::Float64
    mu_safe = max(mu, 0.0)
    h0 = hill_activation(mu_safe, K, n)
    if sigma2 <= 0.0 || mu_safe <= 1e-300
        return h0
    end
    h2 = _hill_activation_d2(mu_safe, K, n)
    return clamp(h0 + 0.5 * h2 * sigma2, 0.0, 1.0)
end

"""
    _hill_activation_d2(x, K, n)

Second derivative of hill_activation(x, K, n) = x^n/(x^n + K^n):

    h″(x) = n K^n x^{n-2} / (x^n + K^n)³ × [(n-1)K^n − (n+1)x^n]
"""
function _hill_activation_d2(x::Float64, K::Float64, n::Float64)::Float64
    if x <= 1e-300
        if n > 2.0
            return 0.0
        elseif abs(n - 2.0) < 0.01
            return 2.0 / K^2
        elseif abs(n - 1.0) < 0.01
            return -2.0 / K^2
        else
            return 0.0
        end
    end
    xn = x^n
    Kn = K^n
    D = xn + Kn
    return n * Kn * x^(n - 2.0) / D^3 * ((n - 1.0) * Kn - (n + 1.0) * xn)
end
