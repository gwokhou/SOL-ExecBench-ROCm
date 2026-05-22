# Phase 27 Summary: AMD SOL Analyzer Coverage

**Completed:** 2026-05-22
**Status:** Complete
**Requirements:** SOLCOV-01, SOLCOV-02, SOLCOV-03, SOLCOV-04

## What Changed

- Added a derived AMD SOL coverage summary to bound artifacts, including
  supported, inexact, unsupported, and per-op-type counts.
- Broadened analyzer recognition beyond v1.5 matmul/elementwise to include
  reductions, normalization-like calls, softmax-like calls, activation calls,
  and data-movement/view-like operations.
- Added conservative FLOP/byte estimates for the new recognized operation
  families while preserving unsupported operations as first-class evidence.
- Documented AMD SOL coverage semantics and confidence labels.
- Added tests for coverage summaries, source confidence, softmax/data movement,
  unsupported preservation, and canonical trace immutability.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_sol_bounds.py` - passed
- `uv run ruff check src/sol_execbench/core/scoring/amd_sol.py tests/sol_execbench/test_amd_sol_bounds.py` - passed

## Compatibility

Canonical trace JSONL and public trace schemas were not changed. New coverage
data is emitted only through derived AMD SOL artifacts.
