# Phase 134 Summary: NVIDIA/Blackwell Low-Precision ROCm Equivalence

**Completed:** 2026-06-04
**Status:** Complete

## Delivered

- Added CPU-safe low-precision compatibility helpers for `nvfp4`, `mxfp4`,
  `float4_e2m1`, and `float4_e2m1fn_x2`.
- Implemented E2M1 quantize/dequantize and final-dimension two-values-per-byte
  pack/unpack helpers with original shape preservation.
- Added scale metadata and explicit unvalidated-CDNA4 compatibility evidence
  models.
- Exported the compatibility surface from the dataset package.
- Integrated Phase 134 evidence markers into Blackwell/NVFP4 readiness
  classification while preserving `needs_hardware_evidence` blockers.
- Added CPU-safe tests for round trips, scalar behavior, fallback/evidence
  markers, validation errors, and readiness blocker reporting.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_low_precision_compatibility.py tests/sol_execbench/test_dataset_inventory_readiness.py`
  - `25 passed`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/dataset/low_precision.py src/sol_execbench/core/dataset/readiness.py src/sol_execbench/core/dataset/__init__.py tests/sol_execbench/test_low_precision_compatibility.py tests/sol_execbench/test_dataset_inventory_readiness.py`
  - passed
- `git diff --check`
  - passed

## Deferred

- Real CDNA4 hardware validation, performance authority, NVIDIA
  Blackwell/B200 equivalence, paper parity, and score authority remain
  deferred.
- Dataset runner integration and public guardrails remain Phase 135.
