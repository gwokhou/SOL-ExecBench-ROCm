# Phase 30 Summary: Compatibility and Claim Guardrails

**Completed:** 2026-05-22
**Status:** Complete
**Requirements:** COMPAT-01, COMPAT-02, COMPAT-03, COMPAT-04, CLAIM-01, CLAIM-02, CLAIM-03

## What Changed

- Added v1.6 public contract guardrail tests for primary CLI compatibility,
  derived artifact separation, CDNA3 validation deferral, NVIDIA/SOLAR no-claim
  language, and hardware model evidence preservation.
- Updated CDNA3 score warning text so it is not tied to v1.5.
- Updated analysis documentation to state CDNA3 full-suite validation is not
  part of v1.6.

## Verification

- `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py` - passed
- `uv run ruff check src/sol_execbench/core/scoring/amd_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_native_score.py` - passed

## Compatibility

Primary `sol-execbench` CLI defaults, public schemas, canonical trace JSONL,
and derived artifact boundaries remain guarded by tests.
