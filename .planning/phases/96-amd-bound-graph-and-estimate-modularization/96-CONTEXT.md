---
phase: 96
status: active
created: 2026-06-01
---

# Phase 96 Context: AMD Bound Graph And Estimate Modularization

## Goal

Reduce AMD bound graph and estimate module coupling by extracting operation
family classification and estimate family dispatch into focused, testable
helpers while preserving public bound graph and estimate schemas.

## Scope

- `src/sol_execbench/core/scoring/amd_bound_graph.py`
- `src/sol_execbench/core/scoring/amd_bound_estimates.py`
- New helper modules under `src/sol_execbench/core/scoring/`
- Focused tests in `tests/sol_execbench/test_amd_bound_graph.py` and
  `tests/sol_execbench/test_amd_bound_estimates.py`

## Non-Goals

- No changes to canonical benchmark data schemas.
- No score-authority or claim-boundary changes.
- No SOLAR derivation or static evidence decomposition; those are Phase 97.
- No formula behavior changes unless existing tests force a compatibility fix.

## Requirements

- ANALYSIS-01: Separate graph construction, operator-family classification, and
  evidence annotation into smaller helpers with operation-family tests.
- ANALYSIS-02: Group AMD bound estimate formulas by family/responsibility so
  one family can be reasoned about without broad scoring regressions.

## Current Shape

`amd_bound_graph.py` owns IR dataclasses, FX/AST extraction, classification
tables, family annotation, and attribute derivation. `amd_bound_estimates.py`
owns estimate dataclasses, dispatch, formulas, byte helpers, and shape inference.

Phase 96 should make the first low-risk cuts: move classification taxonomy out
of graph construction, and make estimate dispatch table-driven by family group.
