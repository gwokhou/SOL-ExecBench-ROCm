# Phase 35 Plan: MI300X Validation Readiness Guardrails

**Status:** Complete

## Tasks

- [x] Add a pure MI300X validation evidence gate that reports blockers when
  hardware/environment/full-suite/clock/artifact evidence is incomplete.
- [x] Document MI300X validation commands, required hardware, ROCm stack,
  clock-lock setup, artifacts, and acceptance criteria.
- [x] Document FP8 readiness for future MI300X execution and keep NVFP4/MXFP4
  explicitly deferred.
- [x] Update public no-claim wording to name MI300X as the future commercial GPU
  validation target.
- [x] Add tests for complete and incomplete MI300X validation evidence.
- [x] Add documentation tests for MI300X handoff and no-claim wording.

## Verification

- `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py`
- `uv run ruff check src/sol_execbench/core/diagnostics.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py`
