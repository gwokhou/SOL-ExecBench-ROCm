# Phase 129 Summary: Deferred-Execution Guardrails

**Completed:** 2026-06-04
**Status:** Complete

## Delivered

- Updated public contract guardrails to read the archived
  `.planning/milestones/CDNA3-VALIDATION-HANDOFF.md` path.
- Strengthened `test_cdna3_validation_remains_deferred_in_docs` to assert v1.28
  readiness remains separate from actual MI300X/gfx942 execution.
- Verified CDNA3 score reports still carry `CDNA3_NO_VALIDATION_WARNING`.
- Verified MI300X claim blockers still reject incomplete evidence after Phase
  128 expanded artifact and result-category requirements.
- Verified Phase 127's AST audit still enforces a concrete `requires_cdna3`
  hardware-gated test surface.

## Verification

- `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py::test_cdna3_validation_remains_deferred_in_docs tests/sol_execbench/test_amd_native_score.py::test_unsupported_cdna3_score_carries_no_validation_guardrails tests/sol_execbench/test_rocm_diagnostics_reporting.py::test_mi300x_validation_claim_requires_complete_evidence tests/sol_execbench/test_rocm_test_suite_audit.py::test_cdna3_marker_has_concrete_hardware_gated_test_surface`
  - `4 passed`
- `uv run --with ruff ruff check tests/sol_execbench/test_public_contract_guardrails.py`
  - passed

## Deferred

- No claim-upgrade path was activated.
- Real CDNA3/MI300X evidence remains deferred.
