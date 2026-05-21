# Phase 04 ROCm Replacement Decisions

## Strategy

Use schema-compatible ROCm examples immediately, and avoid speculative rewrites for NVIDIA-specific DSLs or libraries when no tested local ROCm equivalent exists yet.

## Replacement Matrix

| Original category | Phase 4 replacement | Rationale |
| --- | --- | --- |
| CUDA C++ | HIP/C++ (`hip_cpp`) | Direct source and metadata migration is feasible for examples with simple kernels. |
| CUTLASS GEMM | PyTorch fallback now; evaluate Composable Kernel, rocWMMA, rocBLAS, or hipBLASLt for production | A faithful CK/rocWMMA implementation needs hardware validation and tuning. |
| cuDNN softmax | PyTorch fallback now; evaluate MIOpen or a HIP/Triton softmax kernel for production | MIOpen operator coverage and shape behavior should be validated before presenting a library example. |
| CuTe DSL | PyTorch fallback now; evaluate Composable Kernel, rocWMMA, or HIP/Triton kernels for production | NVIDIA CuTe DSL has no direct ROCm-compatible runtime in this project. |
| cuTile | PyTorch fallback now; evaluate Composable Kernel, rocWMMA, or HIP/Triton kernels for production | cuTile is NVIDIA-specific and should not be represented as available under ROCm. |

## Library Notes

- `rocBLAS`: preferred baseline for dense BLAS-style GEMM examples when a library call is sufficient.
- `hipBLASLt`: preferred for production GEMM examples that need epilogues, layouts, or tunable algorithms.
- `MIOpen`: candidate for neural-network primitives, but use only after confirming operator coverage and tensor layout behavior.
- `Composable Kernel`: candidate for high-performance GEMM and fused kernels; requires shape-specific implementation and validation.
- `rocWMMA`: candidate for wave-level matrix multiply examples; appropriate for lower-level educational kernels.
- `hipCUB`: candidate replacement for CUDA CUB primitives such as reductions, scans, and segmented operations.
- `rocPRIM`: candidate for primitive parallel algorithms when HIP/C++ examples need reductions or scans.
- `rocThrust`: candidate for high-level parallel algorithms where existing examples use Thrust-like APIs.

## Current Phase 4 Outcome

- Public CUDA C++ examples are migrated to ROCm schema metadata and `.hip` source naming.
- Former CUTLASS/cuDNN/CuTe DSL/cuTile examples are kept as runnable PyTorch fallback examples with explicit replacement notes.
- Full ROCm library implementations are deferred until they can be compiled and validated on the Phase 5 hardware matrix.
