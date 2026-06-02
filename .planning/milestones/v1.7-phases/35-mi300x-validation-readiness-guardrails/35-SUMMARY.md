# Phase 35 Summary: MI300X Validation Readiness Guardrails

**Completed:** 2026-05-22
**Status:** Complete
**Requirements:** MI3-01, MI3-02, MI3-03

## What Changed

- Added `mi300x_validation_claim_blockers()` and
  `can_mark_mi300x_hardware_validated()` to require complete MI300X evidence
  before reports can claim hardware validation.
- Added `.planning/MI300X-VALIDATION-HANDOFF.md` and
  `docs/internal/mi300x_validation_readiness.md`.
- Updated README and ROCm docs to name AMD Instinct MI300X (`gfx942`) as the
  future commercial GPU validation target without claiming it is validated.
- Documented FP8 as future MI300X validation scope and NVFP4/MXFP4 as
  `deferred_no_amd_path`.
- Added diagnostics and documentation tests for MI300X evidence gates.
- Restored active requirements wording for deferred real CDNA3 `gfx94*`
  full-suite validation.

## Verification

- `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py` - passed
- `uv run ruff check src/sol_execbench/core/diagnostics.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py` - passed

## Compatibility

No validation claim was upgraded. The new evidence gate is pure and only
permits an MI300X-on-CDNA3 validated status when required evidence is present.
