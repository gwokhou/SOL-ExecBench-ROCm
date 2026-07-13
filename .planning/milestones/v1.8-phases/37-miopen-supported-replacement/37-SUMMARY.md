---
phase: 37
status: complete
completed: 2026-05-22
requirements-completed:
  - MIOPEN-01
  - MIOPEN-02
  - MIOPEN-03
  - MIOPEN-04
---

# Phase 37 Summary

Phase 37 promoted MIOpen to supported ROCm library status for the softmax
replacement path.

## Delivered

- Added `examples/miopen/softmax/` with definition, workloads, reference,
  kernel compatibility file, native MIOpen C++ source, and `solution_miopen.json`.
- Implemented `miopenSoftmaxForward_V2` over float32 `[batch_size, 4096]`
  tensors using `MIOPEN_SOFTMAX_ACCURATE` and
  `MIOPEN_SOFTMAX_MODE_INSTANCE`.
- Added public-contract tests for MIOpen metadata, source consistency, native
  staging, RDNA 4 E2E registration, and support-status documentation.
- Added `requires_rocm_dev` test gating for native extension development
  headers, so RDNA 4 E2E tests do not fail on partial ROCm installations.
- Updated `docs/user/rocm_libraries.md` to classify MIOpen as supported and describe
  operation-specific constraints.

## Validation

- Focused test suite: 20 passed, 1 skipped.
- Ruff check for changed Python test files: passed.

## Remaining Scope

Composable Kernel and rocWMMA remain candidate categories and are handled by
Phases 38 and 39.
