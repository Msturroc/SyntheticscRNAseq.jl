#= ================================================================
   Default algorithm selection.

   Returns the fastest algorithm that hits accuracy thresholds
   validated in experiments/validate_analytical.jl.

   Selection logic (based on analytical validation):
   - G <= 10: BinomialTauLeapFast — best accuracy/speed tradeoff,
     naturally non-negative, StaticArrays + Polyester multithreading
   - G > 10:  CLE — vectorized matmul scales better than per-cell
     loops for large G; StaticArrays max practical G ≈ 20
   - GPU:     CLE on GPU — cuBLAS matmuls dominate at large G

   dt=0.1 is the default for all approximate methods (validated
   to < 5% mean error, < 15% variance error vs SSA).
   ================================================================ =#

"""
    default_algorithm(G; gpu=false, dt=0.1)

Return the fastest simulation algorithm that passes analytical
accuracy thresholds for a network with `G` genes.

Selection:
- `G <= 10, cpu`: `BinomialTauLeapFast(dt)` — StaticArrays + Polyester
- `G > 10, cpu`:  `CLE(dt)` — vectorized matmul, scales to large G
- `gpu=true`:     `CLE(dt)` — cuBLAS matmuls (call with `Val(:gpu)`)

Returns an algorithm instance ready for `simulate(net, alg, kin; ...)`.
"""
function default_algorithm(G::Int; gpu::Bool=false, dt::Float64=0.1)
    if gpu
        return CLE(dt)
    elseif G <= 10
        return BinomialTauLeapFast(dt)
    else
        return CLE(dt)
    end
end
