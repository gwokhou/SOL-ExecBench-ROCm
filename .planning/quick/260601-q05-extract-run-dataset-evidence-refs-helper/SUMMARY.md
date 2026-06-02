---
quick_id: 260601-q05
slug: extract-run-dataset-evidence-refs-helper
status: complete
completed: 2026-06-01
---

# Summary

Completed run-dataset evidence-reference helper extraction.

- Added `sol_execbench.core.dataset.evidence_refs` for safe sidecar stems,
  relative refs, and derived evidence refs/gaps.
- Kept `scripts/run_dataset.py` compatibility aliases and orchestration
  behavior.
- Added focused dataset runner AMD score and execution-closure coverage.

Verification recorded in the plan:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_run_dataset_execution_closure.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check scripts/run_dataset.py src/sol_execbench/core/dataset tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_run_dataset_execution_closure.py`
