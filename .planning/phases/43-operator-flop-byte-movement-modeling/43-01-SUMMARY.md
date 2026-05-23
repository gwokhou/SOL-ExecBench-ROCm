---
phase: 43-operator-flop-byte-movement-modeling
plan: 01
subsystem: scoring
tags: [amd, sol, bound-graph, estimates, formulas]
requires: []
provides:
  - Rich operator work estimate contract
  - estimate_bound_work BoundGraph entry point
  - Explicit unsupported estimate fallback
affects: [scoring, amd-bound-estimates, phase-43]
tech-stack:
  added: []
  patterns: [frozen dataclass estimate contract, JSON-safe estimate serialization]
key-files:
  created:
    - src/sol_execbench/core/scoring/amd_bound_estimates.py
    - tests/sol_execbench/test_amd_bound_estimates.py
  modified:
    - src/sol_execbench/core/scoring/__init__.py
key-decisions:
  - "Keep rich estimates outside canonical benchmark schemas."
  - "Return one zero-valued unsupported estimate for every unsupported graph node."
patterns-established:
  - "OperatorWorkEstimate.to_dict() serializes enums as stable string values."
  - "estimate_bound_work() accepts BoundGraph as the primary Phase 43 API."
requirements-completed: [MODEL-05]
duration: "completed before summary backfill"
completed: 2026-05-23
---

# Phase 43-01: Rich Estimate Contract Summary

**BoundGraph nodes now have a rich per-operator estimate contract with formula, byte bucket, confidence, rationale, and unsupported evidence fields.**

## Performance

- **Duration:** completed before summary backfill
- **Started:** not recorded
- **Completed:** 2026-05-23T09:50:19+08:00
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added frozen `OperatorWorkEstimate` with formula fields, byte buckets, confidence, rationale, axis/movement evidence, warnings, and JSON-safe serialization.
- Added `estimate_bound_work(graph)` returning exactly one rich estimate per `BoundGraphNode`.
- Added explicit unsupported fallback for unsupported and out-of-scope operator families.
- Exported `OperatorWorkEstimate` and `estimate_bound_work` through `sol_execbench.core.scoring`.

## Task Commits

1. **Rich estimate contract and API** - `566c15e` (feat)

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - Rich estimate dataclass, byte helpers, graph entry point, and unsupported fallback.
- `src/sol_execbench/core/scoring/__init__.py` - Public scoring exports.
- `tests/sol_execbench/test_amd_bound_estimates.py` - Serialization, export, dtype byte, one-estimate-per-node, and unsupported-family tests.

## Decisions Made

Followed the plan: rich evidence is additive and internal to scoring, while public schemas and CLI behavior remain unchanged.

## Deviations from Plan

Summary creation was backfilled after starting 43-02 execution. Code scope and verification matched the plan.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x` - passed, 6 tests.

## Next Phase Readiness

The contract and dispatcher are ready for formula implementations in later Phase 43 plans.

---
*Phase: 43-operator-flop-byte-movement-modeling*
*Completed: 2026-05-23*
