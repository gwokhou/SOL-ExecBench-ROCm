# Phase 44-01 Summary: V2 Sidecar Contract And Loader

**Status:** Complete
**Completed:** 2026-05-23

## Implemented

- Added `AMD_SOL_V2_SCHEMA_VERSION` and frozen v2 dataclasses in
  `src/sol_execbench/core/scoring/amd_sol_v2.py`.
- Added strict `amd_sol_bound_v2_from_dict()` parsing for schema version,
  required top-level fields, nested op bounds, aggregate bound, coverage
  summary, warnings, and hardware model payloads.
- Added round-trip and malformed payload tests in
  `tests/sol_execbench/test_amd_sol_v2.py`.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` - passed, 7 tests.
- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` - passed, 29 tests.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_sol_v2.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_public_contract_guardrails.py` - passed.

## Requirement Coverage

- BOUND-01: v2 sidecars serialize and load with schema version, derived marker,
  workload identity, graph, estimates, operation bounds, aggregate, hardware
  model reference, warnings, and coverage summary.
