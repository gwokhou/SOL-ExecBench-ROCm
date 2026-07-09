---
status: complete
completed: 2026-07-09
---

# Organize Platform Matrix Modules

## Summary

Grouped the clearest matrix-related `core/platform` feature clusters into
subpackages:

- `dependency_matrix/`
- `docker_matrix/`

Moved their CLI implementations to `cli.py`, retained `python -m` execution via
package `__main__.py`, and updated source, tests, scripts, docs, and planning
references to use the new submodule paths.

## Verification

- `uv run pytest tests/sol_execbench/core/platform/test_dependency_matrix_cli.py tests/sol_execbench/core/platform/test_dependency_matrix_classification.py tests/sol_execbench/core/platform/test_dependency_matrix_policy.py tests/sol_execbench/core/platform/test_docker_matrix_preflight.py tests/sol_execbench/core/platform/test_docker_matrix_targets.py tests/sol_execbench/core/platform/test_run_docker_matrix_script.py tests/sol_execbench/core/reports/test_runtime_evidence_reports.py`
  - 69 passed
- `bash -n scripts/run_docker.sh`
  - passed
- `uv run --with ruff ruff check .`
  - passed
- `uv run ty check`
  - passed
- `uv run pytest tests/`
  - 1885 passed, 41 skipped
