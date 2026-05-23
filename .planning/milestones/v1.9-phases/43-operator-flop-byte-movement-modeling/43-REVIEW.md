---
phase: 43-operator-flop-byte-movement-modeling
status: clean
depth: standard
files_reviewed: 8
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
reviewed_at: 2026-05-23T09:57:32+08:00
---

# Phase 43 Code Review

Review scope:

- `src/sol_execbench/core/scoring/__init__.py`
- `src/sol_execbench/core/scoring/amd_bound_estimates.py`
- `src/sol_execbench/core/scoring/amd_bound_graph.py`
- `src/sol_execbench/core/scoring/amd_sol.py`
- `tests/sol_execbench/test_amd_bound_estimates.py`
- `tests/sol_execbench/test_amd_bound_graph.py`
- `tests/sol_execbench/test_amd_sol_bounds.py`
- `tests/sol_execbench/test_public_contract_guardrails.py`

## Result

No critical, warning, or info findings remain after review.

## Notes

- The rich estimate API is isolated in derived scoring modules and exported deliberately.
- The legacy `amd_sol.estimate_work()` adapter preserves `WorkEstimate` shape and now falls back when caller-provided legacy graph nodes do not align with rebuilt rich graph evidence.
- v1 artifacts and public schema/CLI guardrails prevent rich fields from leaking into canonical contracts.

## Verification Reviewed

- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x`
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/amd_sol.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py`
