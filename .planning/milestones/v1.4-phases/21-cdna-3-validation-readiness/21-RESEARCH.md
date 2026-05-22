# Phase 21 Research: CDNA 3 Validation Readiness

## RESEARCH COMPLETE

**Phase:** 21 - CDNA 3 Validation Readiness  
**Date:** 2026-05-22  
**Mode:** Inline autonomous research from local source code

## Existing Local Assets

- `src/sol_execbench/core/diagnostics.py`
  - `classify_gfx("gfx942") == "cdna3"`
  - `classify_gfx("gfx1200") == "rdna4"`
  - tool diagnostics report `hipcc`, `rocminfo`, `rocm-smi`, and `rocprofv3`
    availability.
- `.planning/CDNA3-VALIDATION-HANDOFF.md`
  - Defines required `gfx94*` hardware, ROCm >= 7.0, full pytest command,
    environment capture, evidence to record, and acceptance criteria.
- `docs/rocm.md`, `docs/solution.md`, and `docs/compliance.md`
  - Explicitly separate CDNA 3 code/schema support from hardware validation.

## Gap For Phase 21

The project has a handoff document and basic diagnostics, but not a reusable
internal readiness object that can:

- distinguish CDNA 3, RDNA 4, unknown, and missing-tool environments;
- list future CDNA 3 validation commands and evidence requirements;
- report blockers without claiming validation;
- be unit-tested without real CDNA 3 hardware.

## Recommended Implementation

- Add a frozen dataclass `ValidationReadiness` and helper
  `cdna3_validation_readiness` to `src/sol_execbench/core/diagnostics.py`.
- Return:
  - `target_family`
  - `ready`
  - `claim`
  - `commands`
  - `evidence_required`
  - `acceptance_criteria`
  - `blockers`
- Use conservative claim strings:
  - "cdna3_readiness_implemented"
  - "cdna3_hardware_validation_deferred"
- Add docs in `docs/internal/cdna3_validation_readiness.md`.
- Add tests in `tests/sol_execbench/test_rocm_diagnostics_reporting.py` and
  `tests/sol_execbench/test_rocm_support_docs.py`.

## Validation Architecture

```bash
uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/diagnostics.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py
```

## Risks

- False support claim: tests must assert readiness wording does not say CDNA 3
  is hardware-validated.
- Overfitting to a current host: helper must accept explicit gfx/tool inputs so
  unit tests do not depend on local hardware.
