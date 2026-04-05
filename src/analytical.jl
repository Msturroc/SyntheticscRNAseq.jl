#= ================================================================
   Analytical solutions for stochastic gene expression models.

   Telegraph model (Peccoud & Ycart 1995, Cao & Grima 2020):
     Exact steady-state mRNA distribution P(n) for the two-state
     promoter model with activation rate λ, deactivation rate μ,
     transcription rate ν (ON only), and mRNA decay rate d.

   Dimensionless parameters: a = λ/d, b = μ/d, c = ν/d.
   ================================================================ =#

using SpecialFunctions: loggamma
using HypergeometricFunctions: _₁F₁

# ── Telegraph model: exact steady-state mRNA distribution ────────

"""
    telegraph_logpmf(n, a, b, c)

Log-probability log P(n) for the telegraph model steady-state mRNA
distribution (Peccoud & Ycart 1995).

Uses the Kummer-transformed form for numerical stability:

    P(n) = exp(-c) * c^n / n! * (a)_n / (a+b)_n * ₁F₁(b; a+b+n; c)

where (a)_n is the Pochhammer symbol (rising factorial).

# Arguments
- `n::Int`: mRNA copy number (≥ 0)
- `a::Real`: dimensionless activation rate λ/d
- `b::Real`: dimensionless deactivation rate μ/d
- `c::Real`: dimensionless transcription rate ν/d
"""
function telegraph_logpmf(n::Int, a::Real, b::Real, c::Real)
    # log P(n) = -c + n*log(c) - loggamma(n+1)
    #          + loggamma(a+n) - loggamma(a)
    #          - loggamma(a+b+n) + loggamma(a+b)
    #          + log(₁F₁(b, a+b+n, c))
    lp = -c + n * log(c) - loggamma(n + 1) +
         loggamma(a + n) - loggamma(a) -
         loggamma(a + b + n) + loggamma(a + b)

    # ₁F₁(b, a+b+n, c) with c > 0 has all-positive series terms
    hyp = _₁F₁(b, a + b + n, c)
    if hyp > 0
        lp += log(hyp)
    else
        # Fallback to log-space series for extreme parameters
        lp += _log_1F1_series(b, a + b + n, c)
    end
    return lp
end

"""
    telegraph_pmf(n, a, b, c)

Probability P(n) for the telegraph model steady-state mRNA distribution.
"""
telegraph_pmf(n::Int, a::Real, b::Real, c::Real) = exp(telegraph_logpmf(n, a, b, c))

"""
    telegraph_distribution(a, b, c; nmax=nothing, tol=1e-12)

Full probability vector P(0), P(1), ..., P(nmax) for the telegraph
model. If `nmax` is not given, automatically determines truncation
where P(n) < tol * max(P).

Returns a Vector{Float64} of length nmax+1 (0-indexed via 1-based array).
"""
function telegraph_distribution(a::Real, b::Real, c::Real;
                                 nmax::Union{Nothing,Int}=nothing,
                                 tol::Float64=1e-12)
    # Auto-determine nmax from mean + generous tail
    if nmax === nothing
        μ = telegraph_mean(a, b, c)
        σ = sqrt(telegraph_variance(a, b, c))
        nmax = max(ceil(Int, μ + 10 * σ), 100)
    end

    logp = [telegraph_logpmf(n, a, b, c) for n in 0:nmax]
    logp_max = maximum(logp)

    # Truncate where probability is negligible
    cutoff = log(tol) + logp_max
    last_significant = findlast(lp -> lp > cutoff, logp)
    if last_significant !== nothing && last_significant < length(logp)
        logp = logp[1:last_significant]
    end

    # Convert to probabilities and normalize (should already sum to ~1)
    p = exp.(logp .- logp_max)
    p ./= sum(p)
    return p
end

"""
    telegraph_mean(a, b, c)

Exact mean mRNA count: E[m] = a*c / (a+b).
"""
telegraph_mean(a::Real, b::Real, c::Real) = a * c / (a + b)

"""
    telegraph_variance(a, b, c)

Exact mRNA variance: Var(m) = mean + a*b*c² / ((a+b)² * (a+b+1)).
"""
function telegraph_variance(a::Real, b::Real, c::Real)
    μ = telegraph_mean(a, b, c)
    return μ + a * b * c^2 / ((a + b)^2 * (a + b + 1))
end

"""
    telegraph_fano(a, b, c)

Exact mRNA Fano factor: F = 1 + b*c / ((a+b) * (a+b+1)).
"""
telegraph_fano(a::Real, b::Real, c::Real) = 1.0 + b * c / ((a + b) * (a + b + 1))

# ── Log-space ₁F₁ series (fallback for overflow) ────────────────

"""
    _log_1F1_series(a, b, z; maxterms=20000, tol=1e-15)

Compute log(₁F₁(a, b, z)) via log-sum-exp accumulation.
Only valid when all series terms are positive (z > 0, a > 0, b > a).
"""
function _log_1F1_series(a::Real, b::Real, z::Real;
                          maxterms::Int=20000, tol::Float64=1e-15)
    # ₁F₁(a,b,z) = Σ_{k=0}^∞ (a)_k / (b)_k * z^k / k!
    # term_k = term_{k-1} * (a+k-1) / (b+k-1) * z / k
    log_term = 0.0  # log of term_0 = 1
    log_sum = 0.0   # log of partial sum starting at 1

    for k in 1:maxterms
        log_term += log(a + k - 1) - log(b + k - 1) + log(z) - log(k)

        # log_sum = log(exp(log_sum) + exp(log_term))
        if log_term > log_sum
            log_sum = log_term + log1p(exp(log_sum - log_term))
        else
            log_sum = log_sum + log1p(exp(log_term - log_sum))
        end

        # Converged when new term is negligible relative to sum
        if log_sum - log_term > -log(tol)
            return log_sum
        end
    end
    return log_sum
end
