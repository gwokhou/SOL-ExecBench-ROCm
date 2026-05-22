---
phase: 40
status: complete
completed: 2026-05-22
---

# Phase 40 Summary

Phase 40 closed the v1.8 compatibility and support-claim cleanup.

## Delivered

- README now lists MIOpen, Composable Kernel, and rocWMMA as scoped supported
  native examples instead of candidate categories.
- ROCm setup docs now point each native library category to its supported public
  example and keep v1.8 validation scoped to RDNA 4.
- ROCm library readiness docs now tie support claims to runnable examples,
  operation scope, and dependency requirements.
- Public-contract tests protect compatibility examples, supported example
  coverage, README wording, and RDNA 4/CDNA deferral claims.

## Validation

- Focused library/docs/example suite: 38 passed, 3 skipped.
- Ruff check for changed Python test files: passed.

## Supported v1.8 Library Examples

- hipBLAS: `examples/hipblas/gemm/`
- MIOpen: `examples/miopen/softmax/`
- Composable Kernel: `examples/ck/gemm/`
- rocWMMA: `examples/rocwmma/gemm/`

CDNA 3 and CDNA 4 validation remains deferred.
