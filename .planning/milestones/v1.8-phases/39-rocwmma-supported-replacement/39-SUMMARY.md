---
phase: 39
status: complete
completed: 2026-05-22
requirements-completed:
  - WMM-01
  - WMM-02
  - WMM-03
  - WMM-04
---

# Phase 39 Summary

Phase 39 promoted rocWMMA to supported ROCm library status for an RDNA 4
matrix-core GEMM path.

## Delivered

- Added `examples/rocwmma/gemm/` with definition, workloads, reference, kernel
  compatibility file, native HIP source, and `solution_rocwmma.json`.
- Implemented a measured native path using `rocwmma::fragment`,
  `rocwmma::load_matrix_sync`, `rocwmma::mma_sync`, and
  `rocwmma::store_matrix_sync`.
- Added public-contract tests for rocWMMA metadata, source consistency, native
  staging, RDNA 4 E2E registration, and support-status documentation.
- Added `requires_rocwmma` test gating for environments missing rocWMMA headers.
- Updated `docs/rocm_libraries.md` to classify rocWMMA as supported for the
  RDNA 4 FP16-to-FP32 matrix-core GEMM path and document CDNA deferral.

## Validation

- Focused test suite: 24 passed, 1 skipped.
- Ruff check for changed Python test files: passed.

## Remaining Scope

Phase 40 closes compatibility wording and records the final RDNA 4-only
milestone evidence.
