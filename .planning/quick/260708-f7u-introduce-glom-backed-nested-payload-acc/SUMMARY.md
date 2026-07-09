---
status: complete
date: 2026-07-08
---

# Quick Task 260708-f7u Summary

Added `glom` as a runtime dependency and introduced
`sol_execbench.core.data.path_access` as the project-level wrapper for nested
payload access.

Applied the helper to persisted AMD score sidecar parsing so nested sidecar
payload reads use `path_get` and `path_require` instead of direct chained dict
lookups. The parser's public failure behavior remains unchanged: invalid
persisted sidecars return `None`.

Verification:
- `uv lock --check --cache-dir /tmp/uv-cache`
- `uv run pytest tests/sol_execbench/core/data/test_path_access.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py`
- `uv run --with ruff ruff check pyproject.toml src/sol_execbench/core/data/path_access.py src/sol_execbench/core/scoring/amd_score_sidecar_parsing.py tests/sol_execbench/core/data/test_path_access.py`
