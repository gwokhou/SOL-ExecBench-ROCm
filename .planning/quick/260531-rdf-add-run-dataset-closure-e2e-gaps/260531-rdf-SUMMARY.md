---
quick_id: 260531-rdf
slug: add-run-dataset-closure-e2e-gaps
status: completed
completed: 2026-05-31T14:25:00Z
---

# Quick Task 260531-rdf Summary

## Changes

- Added reusable helpers for synthetic ready-subset and readiness sidecars in
  `tests/examples/test_rocm_cli_paths.py`.
- Added a ROCm e2e regression that records:
  - one real `attempted_passed` workload,
  - one `filtered` workload via `max_workloads_cap`,
  - one `filtered` missing workload via `workload_not_found`,
  - one `not_attempted` readiness-blocked workload.
- Added a ROCm e2e regression proving stale execution-closure provenance forces
  a fresh run instead of reusing an existing passing trace.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/examples/test_rocm_cli_paths.py`
  - Passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/examples/test_rocm_cli_paths.py -q -rs`
  - `7 passed in 33.21s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m requires_rocm -q -rs`
  - `21 passed in 158.84s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with coverage coverage run --source=src/sol_execbench,scripts -m pytest -m requires_rocm -n 0 -q -rs`
  - `21 passed, 1203 deselected in 159.78s`.

## Coverage Notes

Focused coverage after this task:

| File | Coverage |
| --- | ---: |
| `scripts/run_dataset.py` | 63% |
| `src/sol_execbench/core/dataset/execution_closure.py` | 97% |
| `src/sol_execbench/cli/main.py` | 56% |
| `src/sol_execbench/driver/problem_packager.py` | 90% |

Remaining e2e gaps are now mostly environment-sensitive or derived-evidence
paths: `--profile rocprofv3`, `--lock-clocks`, derived AMD score/SOL/SOLAR
sidecars, CLI timeout/compile-failure logging, and Docker/container validation.
