# Phase 104 Plan 01 Summary

**Completed:** 2026-06-01
**Status:** Complete

## Changes

- Added dependency policy guardrails tying declared Docker target policies to
  project dependency declarations in `pyproject.toml` and `uv.lock`.
- Added uniqueness and ROCm target naming checks for Docker dependency policy
  IDs and index metadata.
- Added execution closure provenance tests for sorted sidecar source refs,
  derived sidecar/cache provenance, checksum sensitivity, and order-insensitive
  requested evidence requirement comparison.
- Added marker audit checks for timing opt-in behavior and absence of MI300X or
  CDNA4 validation shortcut markers.
- Updated CDNA3 validation audit assertions to match the current v1.22
  requirement wording.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_execution_closure_contract.py tests/sol_execbench/test_rocm_test_suite_audit.py -q`
  - 22 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_execution_closure_contract.py tests/sol_execbench/test_rocm_test_suite_audit.py`
  - Passed

## Acceptance

- GUARD-01 complete: dependency policy drift is guarded by target/project
  consistency tests.
- GUARD-02 complete: closure provenance tests cover sidecar refs, stale-order
  handling, and manifest/cache provenance.
- GUARD-03 complete: hardware marker tests keep skipped hardware/timing behavior
  explicit and avoid MI300X/CDNA4 overclaim shortcuts.

