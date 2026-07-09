---
status: complete
quick_id: 260709-itw
slug: import-as
---

# Quick Task 260709-itw Summary

Removed redundant `name as name` import aliases reported by Ruff `PLC0414` in:

- `src/sol_execbench/cli/evaluation/command.py`
- `src/sol_execbench/cli/sidecars/common.py`
- `src/sol_execbench/core/bench/input_generation.py`
- `src/sol_execbench/core/dataset/runner_scoring.py`
- `src/sol_execbench/core/scoring/amd_bound_graph/fx_helpers.py`

Verification:

- `uv run --with ruff ruff check . --select PLC0414`
- `uv run --with ruff ruff check src/sol_execbench/cli/evaluation/command.py src/sol_execbench/cli/sidecars/common.py src/sol_execbench/core/bench/input_generation.py src/sol_execbench/core/dataset/runner_scoring.py src/sol_execbench/core/scoring/amd_bound_graph/fx_helpers.py`
- `uv run pytest tests/sol_execbench/cli/test_module_boundaries.py tests/sol_execbench/cli/commands/test_diagnostics.py tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py`
