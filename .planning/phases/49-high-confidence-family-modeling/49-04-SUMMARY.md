---
phase: 49-high-confidence-family-modeling
plan: 04
subsystem: scoring
tags: [solar-derivation, convolution, embedding-positional, public-guardrails]
requires: [49-01, 49-02, 49-03]
provides: [DERIVE-03, DERIVE-05, MODEL-01, MODEL-02, MODEL-05]
affects:
  - src/sol_execbench/core/scoring/amd_bound_graph.py
  - src/sol_execbench/core/scoring/amd_bound_estimates.py
  - src/sol_execbench/core/scoring/solar_derivation.py
tech_stack:
  added: []
  patterns: [BoundGraph attributes, OperatorWorkEstimate, SolarSemanticGroupEvidence]
key_files:
  created:
    - .planning/phases/49-high-confidence-family-modeling/49-04-SUMMARY.md
  modified:
    - src/sol_execbench/core/scoring/amd_bound_graph.py
    - src/sol_execbench/core/scoring/amd_bound_estimates.py
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_amd_bound_graph.py
    - tests/sol_execbench/test_amd_bound_estimates.py
    - tests/sol_execbench/test_solar_derivation_family_modeling.py
    - tests/sol_execbench/test_public_contract_guardrails.py
decisions:
  - Phase 49 convolution and embedding/positional evidence stays internal to SOLAR sidecars.
  - Lookup byte estimates count index bytes plus selected table/output bytes, not dense full-table reads.
  - Public guardrails allow existing AMD SOL v1/v2 bound fields while forbidding SOLAR sidecar fields from canonical schemas and score refs.
metrics:
  duration: 5min
  completed: 2026-05-23T07:14:27Z
---

# Phase 49 Plan 04: Convolution and Memory-Bound Family Modeling Summary

Implemented formula-backed convolution and embedding/positional/gather/rotary-like memory-bound SOLAR evidence while keeping MoE and SSM/Mamba deferred.

## Completed Tasks

| Task | Status | Commit | Summary |
|------|--------|--------|---------|
| 49-04-01 | complete | 023865f, 622dff3 | Added convolution tests and implementation for conv1d/2d/3d metadata, grouped/depthwise formulas, byte evidence, and degraded missing metadata. |
| 49-04-02 | complete | 023865f, 622dff3 | Added embedding, gather, positional, and rotary-like memory-bound recognition with selected-byte estimates and degraded dynamic metadata handling. |
| 49-04-03 | complete | 023865f | Added public contract guardrails proving SOLAR evidence stays sidecar-only while existing AMD SOL bound fields remain allowed. |

## What Changed

- `amd_bound_graph.py` now classifies `conv1d`, `conv2d`, `conv3d`, embedding, gather/index-select/take, positional add, and rotary-like visible elementwise structures.
- `amd_bound_estimates.py` now emits convolution FLOP formulas and memory-bound lookup/positional/rotary byte formulas through `OperatorWorkEstimate`.
- `solar_derivation.py` now creates convolution and embedding-positional semantic subroles and completeness gates, with formula, byte, and bound evidence inside `SolarSemanticGroupEvidence`.
- Tests cover positive and degraded convolution, lookup, positional, rotary-like, deferred MoE/SSM behavior, and public boundary guardrails.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected new test fixtures to use Definition axis names**
- **Found during:** Task 49-04-01 RED verification
- **Issue:** New test shapes used literal integers, but public `Definition` schemas require shape axis names.
- **Fix:** Added named constant axes for kernel and output dimensions.
- **Files modified:** `tests/sol_execbench/test_amd_bound_graph.py`, `tests/sol_execbench/test_solar_derivation_family_modeling.py`
- **Commit:** 023865f

**2. [Rule 1 - Bug] Tightened dynamic embedding completeness**
- **Found during:** Task 49-04-02 GREEN verification
- **Issue:** FX could infer placeholder output shape for dynamic index inputs, which risked selected-byte evidence without explicit index shape.
- **Fix:** Lookup output and selected-element metadata now require explicit index shape.
- **Files modified:** `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`
- **Commit:** 622dff3

## Threat Flags

None.

## Known Stubs

None. Stub scan found only normal empty-list/dict initializers and test assertions.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_family_modeling.py -k "conv or convolution" -n 0 -x` PASS
- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_family_modeling.py -k "embedding or positional or gather or rotary" -n 0 -x` PASS
- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` PASS, 108 passed
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_public_contract_guardrails.py` PASS

## Self-Check: PASSED

- Summary file exists.
- Task commits exist: 023865f, 622dff3.
- No tracked file deletions were introduced.
