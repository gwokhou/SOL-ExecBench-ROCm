# Project Research Summary: v1.8 ROCm Library Ecosystem Completion

## Stack Additions

- MIOpen should be integrated through native HIP/C++ extension sources using
  MIOpen headers and `-lMIOpen`; softmax is the most direct first example.
- Composable Kernel should be integrated as a CK GEMM or fused GEMM example,
  using existing native compile options for include and link flags.
- rocWMMA should be integrated as a matrix-core GEMM example, with RDNA 4
  support constraints documented.

## Feature Table Stakes

- Each of `miopen`, `ck`, and `rocwmma` has a real public example that declares
  that category, uses that library, compiles, and passes on RDNA 4.
- Dependency and Docker checks identify missing library support by name.
- Documentation promotes categories only when runnable evidence exists.
- Former compatibility examples are replaced, retired, or explicitly labeled so
  users cannot mistake them for supported native replacements.

## Watch Out For

- Do not let library examples silently fall back to PyTorch.
- Do not make CDNA 3 or CDNA 4 validation a v1.8 gate.
- Do not add public schema or trace changes unless absolutely necessary.
- Do not present support examples as performance or SOL-score claims without
  separate profiler and baseline evidence.

## Recommended Phase Shape

1. Build plumbing and dependency diagnostics.
2. MIOpen supported replacement.
3. Composable Kernel supported replacement.
4. rocWMMA supported replacement.
5. Compatibility cleanup and RDNA 4 validation closure.
