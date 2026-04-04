#= ================================================================
   scRNA-seq capture model: LogNormal-Binomial dropout.

   Given true molecule counts X (integer), the observed counts are:
     p_cell ~ LogNormal(log(β) - σ²/2, σ)    (cell-level efficiency)
     Y ~ Binomial(X, p_cell)                  (capture sampling)

   This models both:
   - Cell-to-cell variation in capture efficiency (LogNormal)
   - Molecule-level sampling noise (Binomial)
   ================================================================ =#

"""
    apply_capture(X, capture; rng)

Apply scRNA-seq capture noise to molecule counts.

Arguments:
- `X`: Matrix (cell_num x G) of integer molecule counts
- `capture`: CaptureModel with efficiency and variation parameters
- `rng`: random number generator

Returns Matrix (cell_num x G) of observed counts (integers).
"""
function apply_capture(X::Matrix{<:Real}, capture::CaptureModel;
                       rng::AbstractRNG=Random.default_rng())
    cell_num, G = size(X)

    # Per-cell capture efficiency: LogNormal with mean = capture.efficiency
    mu_log = log(capture.efficiency) - capture.efficiency_std^2 / 2
    sigma_log = capture.efficiency_std

    Y = similar(X, Int)
    for c in 1:cell_num
        # Sample cell-level efficiency
        p_cell = clamp(exp(mu_log + sigma_log * randn(rng)), 0.0, 1.0)
        for g in 1:G
            n_molecules = max(round(Int, X[c, g]), 0)
            if n_molecules == 0 || p_cell == 0.0
                Y[c, g] = 0
            else
                Y[c, g] = rand(rng, Distributions.Binomial(n_molecules, p_cell))
            end
        end
    end
    return Y
end
