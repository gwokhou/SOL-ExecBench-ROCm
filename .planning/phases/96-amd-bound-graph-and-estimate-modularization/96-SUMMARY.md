---
phase: 96
status: complete
completed: 2026-06-01
---

# Phase 96 Summary: AMD Bound Graph And Estimate Modularization

## Completed

- Added `sol_execbench.core.scoring.amd_bound_classification` for
  operation-family call classification, movement-kind classification, and dtype
  method target helpers.
- Updated `amd_bound_graph.py` to delegate call taxonomy decisions to the new
  helper while preserving `OpFamily` node values and serialized graph payloads.
- Added `sol_execbench.core.scoring.amd_bound_estimate_families` for explicit
  estimate dispatch groups by operation family.
- Updated `amd_bound_estimates.py` to use table-driven estimate family dispatch
  while preserving existing formula implementations and outputs.
- Added focused tests for classification helpers and estimate dispatch groups.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_bound_graph.py -q`  
  Result: 26 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -q`  
  Result: 32 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_v2.py -q`  
  Result: 27 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_classification.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/amd_bound_estimate_families.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py`  
  Result: passed.

## Notes

This phase deliberately keeps graph extraction and formula bodies in their
current modules. It carves out taxonomy and dispatch responsibilities first so
future family-specific edits can target smaller helpers without changing public
sidecar schemas.
