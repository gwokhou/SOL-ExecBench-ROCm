---
status: complete
date: 2026-07-08
---

# Quick Task 260708-dwm Summary

Tightened coupling boundaries by making the cross-domain import allowlist carry
per-edge rationale text in the boundary test.

Moved persisted AMD score sidecar parsing helpers from
`amd_score_reports.py` into `amd_score_sidecar_parsing.py`, keeping score report
orchestration focused on dataset/report assembly.

Verification:
- `uv run pytest tests/sol_execbench/test_cli_module_boundaries.py`
- `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py`
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_score_reports.py src/sol_execbench/core/scoring/amd_score_sidecar_parsing.py tests/sol_execbench/test_cli_module_boundaries.py`
