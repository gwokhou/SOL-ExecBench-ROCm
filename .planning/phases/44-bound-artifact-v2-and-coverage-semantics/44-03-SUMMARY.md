# Phase 44-03 Summary: Coverage And Warning Semantics

**Status:** Complete
**Completed:** 2026-05-23

## Implemented

- Added family-aware v2 coverage summaries with total, supported, inexact, and
  unsupported counts.
- Added `op_family_counts`, `confidence_counts_by_family`, and
  `worst_confidence` using `unsupported > inexact > supported`.
- Added deterministic warning generation from graph warnings, estimate
  warnings, operator confidence, hardware/model validation status, and
  aggregate status.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` - passed, 7 tests.
- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` - passed, 29 tests.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_sol_v2.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_public_contract_guardrails.py` - passed.

## Requirement Coverage

- BOUND-03: coverage summaries report supported, inexact, and unsupported
  operation counts by family plus worst confidence.
- BOUND-04: warning prefixes make degraded and unscored aggregate states
  visible to callers.
