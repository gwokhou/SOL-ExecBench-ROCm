---
status: complete
phase: 109
completed: 2026-06-01
---

# Phase 109 Summary

## Completed

- Added `emit_trace_jsonl()` to `eval_runtime.py` for strict JSONL trace
  emission.
- Added `run_reward_hack_check()` to `eval_runtime.py` for reward-hack
  exception routing.
- Updated `eval_driver.py` to use the new helpers while keeping trace assembly
  and staged orchestration local.
- Added the new helpers to the integrity snapshot critical-name list.
- Added focused eval-runtime tests for strict emission and reward-hack routing.

## Verification

- `uv run pytest tests/sol_execbench/core/bench/test_eval_runtime.py tests/sol_execbench/driver/test_eval_driver.py -q`
- `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -q`
- `uv run ruff check src/sol_execbench/core/bench/eval_runtime.py src/sol_execbench/driver/templates/eval_driver.py tests/sol_execbench/core/bench/test_eval_runtime.py`
