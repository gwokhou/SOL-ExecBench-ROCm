---
status: complete
completed: 2026-07-09
---

# Organize Runtime Evidence Modules

## Summary

Grouped the `core/evidence/runtime_evidence*` feature cluster into a
`runtime_evidence/` subpackage:

- `__init__.py`
- `__main__.py`
- `builders.py`
- `collectors.py`
- `cli.py`
- `io.py`
- `models.py`

Updated source, tests, scripts, docs, and planning references to use the new
submodule paths. Preserved `python -m sol_execbench.core.evidence.runtime_evidence`
execution through package `__main__.py`.

## Verification

- `uv run pytest tests/sol_execbench/core/reports/test_runtime_evidence_reports.py tests/sol_execbench/core/evidence/test_public_import_facades.py tests/sol_execbench/core/platform/test_dependency_matrix_cli.py tests/sol_execbench/core/platform/test_docker_matrix_targets.py`
  - 28 passed
- `bash -n scripts/run_docker.sh`
  - passed
- `uv run python -m sol_execbench.core.evidence.runtime_evidence --help`
  - passed
- `uv run --with ruff ruff check .`
  - passed
- `uv run ty check`
  - passed
- `uv run pytest tests/`
  - 1885 passed, 41 skipped
