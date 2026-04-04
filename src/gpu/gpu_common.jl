#= ================================================================
   GPU common utilities.

   3D tensor layout: (G, cell_num, N) where N is the batch dimension.
   This layout is optimal for cuBLAS strided batched GEMM.

   These functions are loaded via the CUDA package extension.
   ================================================================ =#

# Placeholder — actual implementations in ext/SyntheticscRNAseqCUDAExt.jl
