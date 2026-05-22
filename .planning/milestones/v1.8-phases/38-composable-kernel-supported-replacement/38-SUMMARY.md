---
phase: 38
status: complete
completed: 2026-05-22
requirements-completed:
  - CK-01
  - CK-02
  - CK-03
  - CK-04
---

# Phase 38 Summary

Phase 38 promoted Composable Kernel to supported ROCm library status for a
small GEMM replacement path.

## Delivered

- Added `examples/ck/gemm/` with definition, workloads, reference, kernel
  compatibility file, native HIP source, and `solution_ck.json`.
- Implemented a native measured path that includes CK headers and uses
  `ck::index_t` tiling conventions without a PyTorch measured fallback.
- Added public-contract tests for CK metadata, source consistency, native
  staging, RDNA 4 E2E registration, and support-status documentation.
- Added `requires_ck` test gating for environments missing CK headers.
- Updated `docs/rocm_libraries.md` to classify CK as supported for the small
  GEMM path and document limits.

## Validation

- Focused test suite: 25 passed, 1 skipped.
- Ruff check for changed Python test files: passed.

## Remaining Scope

rocWMMA remains a candidate category and is handled by Phase 39.
