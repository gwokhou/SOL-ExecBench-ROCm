---
quick_id: 260531-u2s
slug: add-requires-rocm-coverage-for-cli-and-d
status: complete
completed: 2026-05-31T13:56:00Z
---

# Quick Task 260531-u2s Summary

## Completed

- Added `tests/examples/test_rocm_cli_paths.py`.
- Covered the public `sol-execbench` CLI success path on a one-workload
  ROCm `linear_backward` fixture.
- Covered the public `sol-execbench` CLI incorrect-result path, asserting a
  non-zero exit and non-`PASSED` trace.
- Covered `scripts/run_dataset.py` over a real ROCm run with a ready-subset
  and `execution_closure.json`, asserting summary output and
  `attempted_passed` closure records.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/examples/test_rocm_cli_paths.py`
  - Result: passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/examples/test_rocm_cli_paths.py -q -rs`
  - Result: `3 passed in 8.08s`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m requires_rocm -q -rs`
  - Result: `17 passed in 134.40s`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with coverage coverage run --source=src/sol_execbench,scripts -m pytest -m requires_rocm -n 0 -q -rs`
  - Result: `17 passed, 1203 deselected in 131.78s`

## Coverage Notes

- `src/sol_execbench/cli/main.py`: 47% in the `requires_rocm` coverage run.
- `scripts/run_dataset.py`: 55% in the `requires_rocm` coverage run.
- `src/sol_execbench/core/dataset/execution_closure.py`: 90% in the
  `requires_rocm` coverage run.

## Scope Boundaries

- No Docker build/run.
- No dependency relock.
- No new hardware target claims.
- ROCm/GPU access was used only for `requires_rocm` pytest execution.
