#= ================================================================
   Validation utilities: KS tests, moment comparison, convergence.
   ================================================================ =#

using Statistics

"""
    ks_statistic(x, y)

Two-sample Kolmogorov-Smirnov statistic (maximum absolute difference
between empirical CDFs). Does not compute p-value — use for comparing
distributions from different algorithms.
"""
function ks_statistic(x::AbstractVector, y::AbstractVector)
    nx, ny = length(x), length(y)
    sx = sort(x)
    sy = sort(y)

    # Merge and walk both CDFs
    ix, iy = 1, 1
    d_max = 0.0
    ecdf_x, ecdf_y = 0.0, 0.0

    while ix <= nx || iy <= ny
        if ix > nx
            ecdf_y += 1.0 / ny
            iy += 1
        elseif iy > ny
            ecdf_x += 1.0 / nx
            ix += 1
        elseif sx[ix] <= sy[iy]
            ecdf_x += 1.0 / nx
            ix += 1
        else
            ecdf_y += 1.0 / ny
            iy += 1
        end
        d_max = max(d_max, abs(ecdf_x - ecdf_y))
    end
    return d_max
end

"""
    ks_critical_value(n1, n2; alpha=0.05)

Approximate critical value for the two-sample KS test at significance
level alpha. Reject H0 (same distribution) if D > critical value.

Uses the asymptotic formula: c(α) * sqrt((n1+n2)/(n1*n2))
"""
function ks_critical_value(n1::Int, n2::Int; alpha::Float64=0.05)
    # c(α) values: 0.10 → 1.22, 0.05 → 1.36, 0.01 → 1.63
    c_alpha = if alpha <= 0.01
        1.63
    elseif alpha <= 0.05
        1.36
    else
        1.22
    end
    return c_alpha * sqrt((n1 + n2) / (n1 * n2))
end

"""
    compare_moments(samples1, samples2; rtol=0.05)

Compare mean and variance of two sample sets.
Returns (mean_relerr, var_relerr, pass) where pass is true
if both relative errors are below rtol.
"""
function compare_moments(x::AbstractVector, y::AbstractVector; rtol::Float64=0.05)
    m1, m2 = mean(x), mean(y)
    v1, v2 = var(x), var(y)
    mean_err = abs(m1 - m2) / max(abs(m1), abs(m2), 1e-10)
    var_err = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-10)
    return (mean_relerr=mean_err, var_relerr=var_err,
            pass=mean_err < rtol && var_err < rtol)
end

"""
    convergence_rate(dts, errors)

Estimate convergence rate by fitting log(error) = a + rate * log(dt).
Returns the estimated rate (should be ~1.0 for Euler, ~2.0 for midpoint).
"""
function convergence_rate(dts::Vector{Float64}, errors::Vector{Float64})
    log_dt = log.(dts)
    log_err = log.(max.(errors, 1e-16))
    # Simple linear regression: log_err = a + rate * log_dt
    n = length(dts)
    x_bar = mean(log_dt)
    y_bar = mean(log_err)
    rate = sum((log_dt .- x_bar) .* (log_err .- y_bar)) /
           sum((log_dt .- x_bar) .^ 2)
    return rate
end
