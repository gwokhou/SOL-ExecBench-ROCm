# Phase 44-02 Summary: V2 Builder And Aggregate Semantics

**Status:** Complete
**Completed:** 2026-05-23

## Implemented

- Added `build_amd_sol_bound_v2_artifact()` over `Definition`, `Workload`,
  `AmdHardwareModel`, `build_bound_graph()`, and `estimate_bound_work()`.
- Added per-operation v2 bounds derived from rich `OperatorWorkEstimate.flops`
  and `OperatorWorkEstimate.total_bytes`.
- Added aggregate bound states:
  - `scored` for fully supported/validated evidence.
  - `degraded` for inexact or provisional evidence.
  - `unscored` for unsupported or missing operation bound evidence.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` - passed, 7 tests.
- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` - passed, 29 tests.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_sol_v2.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_public_contract_guardrails.py` - passed.

## Requirement Coverage

- BOUND-02: every operation reports compute bound, memory bound, SOL bound,
  limiting resource, confidence, and rationale.
- BOUND-04: unsupported operation evidence forces `unscored` aggregate state
  and cannot silently improve the score signal.
