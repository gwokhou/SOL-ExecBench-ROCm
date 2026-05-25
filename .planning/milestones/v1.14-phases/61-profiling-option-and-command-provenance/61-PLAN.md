# Phase 61 Plan

## Objectives

- Implement `--profile {none,rocprofv3}` with default `none`.
- Build `rocprofv3` diagnostic commands with explicit provenance.
- Detect missing profiler tools before collection and route to a nonfatal
  unavailable state.

## Tasks

- [x] Add profiler profile request/result models and command builder.
- [x] Add CLI profile option and normal-execution fallback.
- [x] Add tests for command shape, unavailable state, and public CLI help.

## Verification

- `uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py -q`
- `uv run --with ruff ruff check ...`
