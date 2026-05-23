# Phase 44-04 Summary: Public Exports And Compatibility Guardrails

**Status:** Complete
**Completed:** 2026-05-23

## Implemented

- Exported v2 scoring APIs from `src/sol_execbench/core/scoring/__init__.py`.
- Added smoke tests that public exports resolve to the v2 module definitions.
- Added guardrails proving v1 `AmdSolBoundArtifact` does not emit v2-only
  fields.
- Extended primary CLI and canonical schema guardrails for v2 sidecar-only
  fields and options.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` - passed, 7 tests.
- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` - passed, 29 tests.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_sol_v2.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_public_contract_guardrails.py` - passed.

## Requirement Coverage

- BOUND-01 through BOUND-04: v2 APIs are programmatic and deliberate while v1
  artifacts, primary CLI help, and canonical data schemas remain unchanged.
