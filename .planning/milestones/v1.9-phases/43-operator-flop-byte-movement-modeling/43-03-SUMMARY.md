---
phase: 43-operator-flop-byte-movement-modeling
plan: 03
subsystem: scoring
tags: [amd, sol, axis, softmax, reduction, normalization, movement]
requires:
  - phase: 43-01
    provides: Rich operator estimate contract and estimate_bound_work API
provides:
  - Axis metadata extraction through BoundGraphNode.attributes
  - Reduction, normalization, and softmax conservative estimates
  - Logical view, broadcast, materialized movement, and dtype conversion byte evidence
affects: [scoring, amd-bound-graph, amd-bound-estimates, phase-43]
tech-stack:
  added: []
  patterns: [attribute-based estimator evidence, conservative pass-count formulas]
key-files:
  created: []
  modified:
    - src/sol_execbench/core/scoring/amd_bound_graph.py
    - src/sol_execbench/core/scoring/amd_bound_estimates.py
    - tests/sol_execbench/test_amd_bound_graph.py
    - tests/sol_execbench/test_amd_bound_estimates.py
key-decisions:
  - "Axis, dtype, and movement hints remain in BoundGraphNode.attributes rather than new dataclass fields."
  - "Reduction, normalization, and softmax estimates are conservative and inexact."
  - "Logical view and broadcast evidence record zero movement bytes; materialized movement and dtype conversion count movement traffic."
patterns-established:
  - "FX attributes merge trace_source with estimator metadata."
  - "AST and FX extraction both record dim/axis, target_dtype, and movement_kind when literal evidence is available."
requirements-completed: [MODEL-03, MODEL-04, MODEL-05]
duration: "not recorded"
completed: 2026-05-23
---

# Phase 43-03: Axis And Movement Estimate Summary

**Reduction, normalization, softmax, data movement, and dtype conversion nodes now expose conservative formula and byte evidence from graph attributes.**

## Performance

- **Duration:** not recorded
- **Started:** not recorded
- **Completed:** 2026-05-23T09:50:19+08:00
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added FX and AST attribute extraction for `dim`/axis, `target_dtype`, and `movement_kind` without expanding `BoundGraphNode`.
- Implemented conservative inexact formulas for reduction, normalization, and softmax.
- Implemented zero-movement logical/broadcast view evidence and nonzero materialized/dtype-conversion movement evidence.
- Added golden tests covering axis metadata, graph contract stability, pass-count estimates, view movement, contiguous movement, and dtype conversion bytes.

## Task Commits

1. **Axis and movement estimates** - `3e1a7bc` (feat)

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_bound_graph.py` - Attribute extraction for axis, dtype, and movement evidence.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - Estimates for reduction, normalization, softmax, data movement, and dtype conversion.
- `tests/sol_execbench/test_amd_bound_graph.py` - Attribute metadata and dataclass-contract guard tests.
- `tests/sol_execbench/test_amd_bound_estimates.py` - Golden estimator tests for axis, pass-count, movement, and conversion evidence.

## Decisions Made

Followed the plan: metadata is evidence stored in `attributes`; all pass-count formulas are conservative and inexact; materialized movement is distinct from logical view evidence.

## Deviations from Plan

None - plan executed as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py -x` - passed, 28 tests.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py` - passed.

## Next Phase Readiness

Rich estimates are ready to be adapted back into the legacy `amd_sol.WorkEstimate` compatibility view.

---
*Phase: 43-operator-flop-byte-movement-modeling*
*Completed: 2026-05-23*
