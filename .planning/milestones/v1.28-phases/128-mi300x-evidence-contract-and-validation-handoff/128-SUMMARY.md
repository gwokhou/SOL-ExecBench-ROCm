# Phase 128 Summary: MI300X Evidence Contract and Validation Handoff

**Completed:** 2026-06-04
**Status:** Complete

## Delivered

- Expanded `MI300X_REQUIRED_ARTIFACTS` to require per-problem traces, ROCm
  timing evidence, AMD-native score report, FP8 result, and NVFP4/MXFP4
  deferred status in addition to suite, dataset, environment, and clock
  evidence.
- Added `MI300X_VALIDATION_RESULT_CATEGORIES` for expected skips, missing
  tools, functional failures, timing instability, missing evidence, FP8
  validation, and deferred quantization formats.
- Strengthened `mi300x_validation_claim_blockers()` so MI300X validation claims
  require both the expanded artifact list and result-category coverage.
- Updated MI300X/CDNA3 handoff docs under `.planning/milestones/` and internal
  readiness docs to describe command order, artifact gates, result categories,
  FP8 readiness, and NVFP4/MXFP4 deferral.
- Updated support-doc tests to use the archived handoff paths created by the
  prior GSD health cleanup.

## Verification

- `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py`
  - `20 passed`
- `uv run --with ruff ruff check src/sol_execbench/core/diagnostics.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py`
  - passed

## Deferred

- No MI300X command was executed in this phase.
- Actual full-suite hardware validation remains deferred until real `gfx942`
  hardware and a complete evidence archive are available.
