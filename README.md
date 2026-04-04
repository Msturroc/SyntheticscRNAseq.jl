# SyntheticscRNAseq.jl

A pure Julia, GPU-accelerated simulator for generating synthetic scRNA-seq data from stochastic gene regulatory network models.

SyntheticscRNAseq.jl provides seven CPU and two GPU simulation algorithms for the two-stage model of gene expression (transcription and translation) with Hill-function regulation, telegraph promoter switching, cell population dynamics and a scRNA-seq capture/dropout model. All algorithms are validated against exact analytical results from stochastic gene expression theory.

## What it does

The package simulates the full pipeline from gene regulatory network parameters to synthetic scRNA-seq count matrices. A gene network is defined by basal transcription rates, a regulatory interaction matrix and optionally per-gene promoter switching rates (telegraph/bursting model). The simulator propagates mRNA and protein counts for each cell using one of several stochastic algorithms, then applies a LogNormal-Binomial capture model that mimics the molecule sampling and cell-to-cell efficiency variation seen in real 10x Chromium data.

The population dynamics module implements a Moran process with exponential volume growth, volume-dependent transcription and binomial molecule partitioning at cell division. This produces the correct steady-state volume distribution and emergent dilution effects without an explicit dilution rate term, following the framework described in Sturrock and Sturrock (2026).

![Single-cell time series showing telegraph promoter switching, mRNA bursts and protein dynamics](figures/timeseries_bursty.png)

## Telegraph promoter model

Each gene can optionally switch between an ON state (transcribes at rate beta) and an OFF state (silent), with rates k_on and k_off. This produces transcriptional bursting with burst size b = beta/k_off and burst frequency f = k_on. The mRNA Fano factor is F = 1 + b/(1 + k_on/k_off + mu_m/k_off), which in the bursty limit reduces to F = 1 + b. Genes with k_on = k_off = Inf (the default) behave as constitutive.

![Constitutive vs bursty mRNA distributions with the same mean expression level](figures/mrna_distribution.png)

## Population dynamics

When a PopulationConfig is provided, cells grow exponentially in volume and divide when reaching a threshold. At division, molecules are partitioned binomially between mother and daughter, and the daughter replaces a random cell in the population (Moran replacement). Transcription rate scales linearly with cell volume, so larger cells produce more mRNA. There is no explicit dilution term in this mode because dilution is emergent from the division process.

For simulations without explicit population dynamics, the dilution field on KineticParams adds a constant dilution rate to both mRNA and protein decay, giving the analytical steady state from Sturrock and Sturrock (2026): mean mRNA = beta/(mu_m + mu) and protein Fano = 1 + k_t/(mu_m + mu_p + 2*mu).

![Population snapshot coloured by cell volume, showing volume-dependent expression](figures/population_snapshot.png)

## Capture model

The CaptureModel applies a LogNormal-Binomial dropout process to true molecule counts. Each cell receives a capture efficiency drawn from a LogNormal distribution, then each molecule is independently captured with that probability. This reproduces the sparsity, library size variation and overdispersion structure observed in real scRNA-seq data.

![Mean-variance relationship before and after 10 percent capture efficiency](figures/capture_model.png)

## Algorithms

The package provides seven CPU algorithms and two GPU algorithms. All share the same interface through the simulate function and accept identical network, kinetics and population parameters.

| Algorithm | Type | Time (G=5, 5000 cells) | Max mean error | Max variance error |
| --- | --- | --- | --- | --- |
| SSA | Exact (Gillespie) | 29.2s | reference | reference |
| PoissonTauLeap | Tau-leap | 4.0s | 0.9% | 4.8% |
| BinomialTauLeap | Tau-leap | 5.9s | 1.9% | 4.1% |
| MidpointTauLeap | Tau-leap | 5.5s | 0.6% | 4.2% |
| CLE | Langevin SDE | 0.8s | 0.9% | 5.3% |
| CLEFast | Langevin SDE | 0.5s | 0.7% | 3.7% |
| BinomialTauLeapFast | Tau-leap | 4.3s | 1.6% | 3.3% |

The GPU variants (CLE and BinomialTauLeap) are loaded automatically when CUDA.jl is available and provide batched simulation via cuBLAS strided batched GEMM for training data generation.

The default_algorithm function returns BinomialTauLeapFast for networks with 10 or fewer genes and CLE for larger networks. BinomialTauLeapFast uses StaticArrays for compile-time-specialised matmuls and Polyester for multithreading. CLE uses standard BLAS matmuls which scale better for large G.

## Comparison with real data

The experiments directory includes a script that downloads the 10x Genomics PBMC 3K dataset, subsets to CD14+ monocytes and compares summary statistics against synthetic output. With calibrated parameters (basal rates matched to observed gene means, capture efficiency fitted from Fano factors), the simulator reproduces the sparsity and mean expression levels of real scRNA-seq. Overdispersion matching requires the telegraph promoter model, since even within a single cell type the median Fano factor is above 1.

![UMAP embedding of real CD14+ monocytes alongside synthetic scRNA-seq cells](figures/umap_comparison.png)

## Analytical validation

The test suite (183 tests) validates all algorithms against exact results from stochastic gene expression theory.

The two-stage model moments (Thattai and van Oudenaarden 2001) are checked for mRNA mean, mRNA variance, protein mean, protein Fano factor and mRNA-protein covariance across all seven CPU algorithms. The Fano factor relationship F = 1 + k_t/(mu_m + mu_p) is verified across a sweep of translation rates.

The dilution model (Sturrock and Sturrock 2026) is validated by comparing molecule concentrations (counts divided by volume) between the population model and the constant-dilution model. The volume distribution in the Moran population is tested against the analytical prediction: mean volume = V_div/(2 ln 2), inverse volume = 2 ln 2/V_div, and median volume = V_div/sqrt(2).

The Grima LNA breakdown test confirms that BinomialTauLeap produces more accurate Fano factors than the CLE at low molecule counts (mean mRNA around 1), where the Gaussian diffusion approximation fails. At high counts (mean mRNA around 100), the CLE recovers its accuracy.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/marcjwilliams1/SyntheticscRNAseq.jl")
```

For GPU support, also install CUDA.jl:

```julia
Pkg.add("CUDA")
```

## Quick start

```julia
using SyntheticscRNAseq

# Define a 5-gene network with sparse regulation
basals = [2.0, 1.5, 3.0, 2.5, 1.0]
A = zeros(5, 5)
A[1, 2] = 5.0   # gene 2 activates gene 1
A[3, 1] = -4.0  # gene 1 represses gene 3
net = GeneNetwork(basals, A)

# Kinetic parameters with growth-coupled dilution
kin = KineticParams(k_t=2.0, K_d=50.0, n=4.0,
                    mu_m=0.1, mu_p=0.2, dilution=0.03)

# Simulate 1000 cells using the default algorithm
alg = default_algorithm(5)
Y = simulate(net, alg, kin; cell_num=1000, T=300.0, readout=:mrna)

# Apply scRNA-seq capture model
cap = CaptureModel(efficiency=0.1, efficiency_std=0.3, readout=:mrna)
Y_obs = apply_capture(Y, cap)
```

For bursty transcription, provide per-gene switching rates:

```julia
# Telegraph model: k_on=0.02, k_off=0.2 gives burst size = beta/k_off
net_bursty = GeneNetwork(basals, A;
                         k_on=fill(0.02, 5),
                         k_off=fill(0.2, 5))
Y_bursty = simulate(net_bursty, BinomialTauLeap(0.05), kin;
                    cell_num=1000, T=500.0, readout=:mrna)
```

For population dynamics with volume-dependent transcription:

```julia
pop = PopulationConfig(cell_num=1000, growth_rate=0.03,
                       V_div=2.0, V_init=(0.8, 1.2))
Y_pop = simulate(net, BinomialTauLeap(0.1), kin;
                 cell_num=1000, T=300.0, readout=:mrna,
                 population=pop)
```

## Running the tests

```julia
using Pkg
Pkg.test("SyntheticscRNAseq")
```

## References

Thattai M, van Oudenaarden A. Intrinsic noise in gene regulatory networks. PNAS, 2001.

Peccoud J, Ycart B. Markovian modeling of gene product synthesis. Theoretical Population Biology, 1995.

Sturrock M, Sturrock CT. Analytical enrichment formulas for gene expression in growing cell populations. bioRxiv, 2026.

Thomas P, Shahrezaei V. Coordination of gene expression noise with cell size. Cell Systems, 2021.

Grima R. An effective rate equation approach to reaction kinetics in small volumes. Journal of Chemical Physics, 2010.

Chatterjee A, Vlachos DG, Katsoulakis MA. Binomial distribution based tau-leap accelerated stochastic simulation. Journal of Chemical Physics, 2005.
