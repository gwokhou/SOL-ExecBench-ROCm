---
status: complete
date: 2026-07-08
---

# Quick Task 260708-dti Summary

Removed the AMD SOL v1 model/coverage import cycle by making
`AmdSolBoundArtifact` store a precomputed coverage summary supplied by the
builder. The artifact payload remains unchanged.

Updated the cross-domain import allowlist to name the concrete module that
depends on `core.dataset.manifest`.

Verification:
- `uv run pytest tests/sol_execbench/test_cli_module_boundaries.py::test_no_internal_two_node_import_cycles`
- `uv run pytest tests/sol_execbench/test_amd_sol_bounds.py`
- `uv run pytest tests/sol_execbench/test_cli_module_boundaries.py`
