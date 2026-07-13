# Phase 130 Summary: CDNA3 Public Documentation Closure

**Completed:** 2026-06-04
**Status:** Complete

## Delivered

- Updated `docs/user/TESTING.md` with direct CDNA3 marker-test commands, expected
  skip interpretation, and the boundary that marker readiness is not full
  MI300X validation evidence.
- Updated `docs/user/rocm.md` to distinguish CDNA3 schema support, concrete CDNA3
  test readiness, deferred MI300X full-suite validation, and unavailable CDNA4.
- Updated `CONTRIBUTING.md` with future CDNA3 test placement, marker usage, and
  MI300X evidence-chain expectations.
- Added documentation guardrails in `test_rocm_matrix_docs.py` and
  `test_rocm_support_docs.py`.

## Verification

- `uv run pytest tests/sol_execbench/test_rocm_matrix_docs.py tests/sol_execbench/test_rocm_support_docs.py`
  - `13 passed`
- `uv run --with ruff ruff check tests/sol_execbench/test_rocm_matrix_docs.py tests/sol_execbench/test_rocm_support_docs.py`
  - passed

## Deferred

- Actual MI300X/gfx942 validation artifacts remain deferred.
