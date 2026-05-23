---
phase: 42-structured-bound-graph-ir
status: clean
reviewed: 2026-05-23
depth: standard
---

# Phase 42 Code Review

## Scope

- `src/sol_execbench/core/scoring/amd_bound_graph.py`
- `src/sol_execbench/core/scoring/amd_sol.py`
- `src/sol_execbench/core/scoring/__init__.py`
- `tests/sol_execbench/test_amd_bound_graph.py`
- `tests/sol_execbench/test_amd_sol_bounds.py`
- `tests/sol_execbench/test_public_contract_guardrails.py`

## Findings

No blocking correctness, security, or compatibility issues found.

## Notes

- The initial implementation risk was that dynamic tracing only validated
  reference execution and then relied on AST evidence. This was addressed by
  adding a `torch.fx` extraction path before AST fallback.
- The compatibility facade keeps legacy `GraphNode.to_dict()` and
  `AmdSolBoundArtifact` v1 payload shape unchanged.
- Public contract guardrails verify bound graph fields do not leak into
  canonical `Definition`, `Workload`, `Trace`, or primary CLI help.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x`
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_sol.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py`

