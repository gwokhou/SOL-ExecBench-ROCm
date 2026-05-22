# Phase 21: CDNA 3 Validation Readiness - Plan

**Status:** Ready for execution  
**Created:** 2026-05-22  
**Requirements:** VAL-04, VAL-05, VAL-06

## Objective

Implement internal CDNA 3 validation readiness metadata for future real
`gfx94*` runs while explicitly avoiding any CDNA 3 hardware-validation claim.

## Scope

In scope:

- Pure readiness helper in `src/sol_execbench/core/diagnostics.py`.
- Internal documentation of commands, evidence, acceptance criteria, and no-claim
  wording.
- Unit tests for CDNA 3, RDNA 4, unknown, and missing-tool cases.

Out of scope:

- Real CDNA 3 hardware execution.
- Updating support matrices to claim CDNA 3 validation.
- Public CLI changes.

## Threat Model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-21-01: Readiness text is read as a validation pass. | High | Helper and docs use explicit deferred claim strings. |
| T-21-02: Unit tests depend on local hardware. | Medium | Helper accepts explicit gfx/tool diagnostics inputs. |
| T-21-03: Missing ROCm tools are hidden. | Medium | Readiness includes blockers and tool diagnostics. |

## Tasks

### Task 21-01: Add Readiness Helper

**Requirement:** VAL-04, VAL-05  
**Files:** `src/sol_execbench/core/diagnostics.py`

1. Add `ValidationReadiness` dataclass.
2. Add `cdna3_validation_readiness(gfx, tool_diagnostics=None)`.
3. Include commands, evidence requirements, acceptance criteria, blockers, and
   conservative claim wording.

### Task 21-02: Document Readiness

**Requirement:** VAL-05  
**Files:** `docs/internal/cdna3_validation_readiness.md`

1. Document expected future commands.
2. Document evidence requirements and acceptance criteria.
3. State that Phase 21 is readiness only, not hardware validation.

### Task 21-03: Add Unit/Doc Guardrails

**Requirement:** VAL-06  
**Files:**

- `tests/sol_execbench/test_rocm_diagnostics_reporting.py`
- `tests/sol_execbench/test_rocm_support_docs.py`

1. Test CDNA 3 readiness on `gfx942`.
2. Test RDNA 4 and unknown-target blockers.
3. Test missing-tool blockers.
4. Test docs do not claim validation.

## Verification

```bash
uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/diagnostics.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py
```

## Completion Criteria

- CDNA 3 readiness is implemented and unit-tested.
- Readiness output includes commands, evidence, acceptance criteria, and
  blockers.
- Docs and tests keep CDNA 3 hardware validation explicitly deferred.
