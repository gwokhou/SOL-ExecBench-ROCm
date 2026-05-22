# Project Research: Stack Additions for v1.8

**Milestone:** v1.8 ROCm Library Ecosystem Completion
**Scope:** RDNA 4 validation only. CDNA 3 and CDNA 4 validation are deferred.

## Existing Stack

- Python package and CLI remain under `src/sol_execbench/`.
- Native ROCm examples compile through `torch.utils.cpp_extension.load` in
  `src/sol_execbench/driver/templates/build_ext.py`.
- `BuildSpec.languages` already recognizes `hip_cpp`, `hipblas`, `miopen`,
  `ck`, and `rocwmma`.
- Current supported native library category is `hipblas`; MIOpen, CK, and
  rocWMMA are still candidate categories.

## Stack Additions

### MIOpen

- Use MIOpen C API through HIP/C++ extension sources.
- Link with `-lMIOpen` and include `<miopen/miopen.h>`.
- Prefer a softmax or convolution example with a simple tensor contract that
  fits existing definition/workload/reference schemas.
- MIOpen API modules include Convolution, Softmax, BatchNorm, Activation,
  Pooling, Reduction, and experimental LayerNorm/GroupNorm/RoPE modules.
- Softmax is a practical first replacement because MIOpen exposes forward and
  backward APIs and the current repo has a former cuDNN softmax compatibility
  example.

Sources:
- https://rocm.docs.amd.com/projects/MIOpen/en/latest/reference/index.html
- https://rocm.docs.amd.com/projects/MIOpen/en/latest/doxygen/html/group__softmax.html

### Composable Kernel

- Use CK through native HIP/C++ extension sources.
- Link/include handling may need CK include paths and optional library paths
  depending on the installed ROCm package layout.
- Prefer a GEMM or GEMM-with-epilogue example because CK documentation centers
  examples around template-instantiated GEMM and fused element operations.
- CK is best positioned as the replacement direction for CUTLASS and some
  CuTe-style fused matmul examples.

Sources:
- https://rocm.docs.amd.com/projects/composable_kernel/en/latest/
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html

### rocWMMA

- Use rocWMMA as a header-oriented HIP/C++ extension category for
  matrix-core-style mixed precision GEMM.
- Include `<rocwmma/rocwmma.hpp>` or equivalent installed headers.
- Linkage may be header-only for the kernel path, but dependency detection must
  verify headers and any package-provided runtime requirements.
- rocWMMA docs list RDNA `gfx1200` and `gfx1201` as supported architectures, so
  RDNA 4 validation is a valid v1.8 target.

Sources:
- https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html
- https://rocmdocs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html

## Build-System Implications

- `build_ext.py` currently passes user-provided `hip_cflags`, `cflags`, and
  `ld_flags`; v1.8 should make library-specific examples self-contained using
  these fields where possible.
- If common include or link flags are repeated across examples, add a small
  helper or documented pattern rather than changing public solution schema.
- Dependency tests should distinguish missing headers, missing libraries, and
  unsupported RDNA 4 architectures.

## What Not To Add

- Do not add new public CLI flags solely for library category support.
- Do not mutate trace JSONL or solution schema names.
- Do not make CDNA 3 or CDNA 4 validation a v1.8 completion gate.
