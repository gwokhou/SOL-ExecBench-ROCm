---
quick_id: 260531-uki
slug: add-remaining-requires-rocm-e2e-coverage
status: complete
completed: 2026-05-31T14:18:00Z
---

# Quick Task 260531-uki Summary

## Changes

- Added a `requires_rocm` HIP/C++ public CLI regression using `examples/hip_cpp/rmsnorm`.
- Added `sol-execbench --static-evidence auto` assertions for the diagnostic-only sidecar contract.
- Added `scripts/run_dataset.py` coverage for first execution, existing-pass reuse, and explicit `--rerun`.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/examples/test_rocm_cli_paths.py`
  - Passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/examples/test_rocm_cli_paths.py -q -rs`
  - `5 passed in 30.65s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m requires_rocm -q -rs`
  - `19 passed in 156.93s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with coverage coverage run --source=src/sol_execbench,scripts -m pytest -m requires_rocm -n 0 -q -rs`
  - `19 passed, 1203 deselected in 154.67s`.

## Coverage Notes

Focused coverage after this task:

| File | Coverage |
| --- | ---: |
| `src/sol_execbench/cli/main.py` | 56% |
| `scripts/run_dataset.py` | 59% |
| `src/sol_execbench/core/bench/static_kernel_evidence.py` | 82% |
| `src/sol_execbench/driver/problem_packager.py` | 90% |

Remaining uncovered e2e surfaces are lower priority for the current no-new-hardware scope:
timing/profile sidecars, lock-clock failure policy, readiness blocked/not-attempted
closure branches, derived scoring sidecars, and stale-provenance mismatch behavior.
