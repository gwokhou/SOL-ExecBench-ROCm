---
phase: 43-operator-flop-byte-movement-modeling
plan: 02
subsystem: scoring
tags: [amd, sol, gemm, bmm, elementwise, activation]
requires:
  - phase: 43-01
    provides: Rich operator estimate contract and estimate_bound_work API
provides:
  - GEMM and batched GEMM formula estimates
  - Elementwise and activation per-node estimates
  - Metadata-gap downgrade tests
affects: [scoring, amd-bound-estimates, phase-43]
tech-stack:
  added: []
  patterns: [node-local tensor byte accounting, visible confidence downgrade]
key-files:
  created: []
  modified:
    - src/sol_execbench/core/scoring/amd_bound_estimates.py
    - tests/sol_execbench/test_amd_bound_estimates.py
key-decisions:
  - "GEMM/BMM complete shape and dtype evidence can be supported."
  - "Elementwise and activation estimates remain inexact because Phase 43 does not prove fusion semantics."
patterns-established:
  - "Formula inputs are derived from node-local BoundTensor shapes."
  - "Missing shape or dtype zeros only the affected bucket and records rationale."
requirements-completed: [MODEL-01, MODEL-02, MODEL-05]
duration: "completed before summary backfill"
completed: 2026-05-23
---

# Phase 43-02: GEMM And Pointwise Estimate Summary

**GEMM, batched GEMM, elementwise, and activation nodes now emit auditable FLOP formulas and node-local read/write byte evidence.**

## Performance

- **Duration:** completed before summary backfill
- **Started:** not recorded
- **Completed:** 2026-05-23T09:50:19+08:00
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Implemented GEMM and batched GEMM estimates with `2*M*N*K` and `2*B*M*N*K` formula evidence.
- Added dtype-aware node-local read/write byte accounting for supported estimate families.
- Added elementwise and activation estimates that remain per-node and inexact.
- Added metadata downgrade tests for missing shape, missing dtype, and fully unresolved known operators.

## Task Commits

1. **GEMM and pointwise estimates** - `ca69cee` (feat)

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - Dispatcher, GEMM/BMM formulas, pointwise formulas, byte helpers, and downgrade logic.
- `tests/sol_execbench/test_amd_bound_estimates.py` - Golden coverage for GEMM, BMM, elementwise/activation chains, and missing metadata.

## Decisions Made

Followed the plan: formula evidence comes from `BoundTensor.shape`; byte evidence comes from node-local tensor IDs; incomplete metadata downgrades confidence instead of fabricating supported work.

## Deviations from Plan

Summary creation was backfilled after the code commit. Code scope and verification matched the plan.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x` - passed, 11 tests.

## Next Phase Readiness

The estimator is ready for axis-aware reduction, normalization, softmax, movement, and dtype conversion evidence.

---
*Phase: 43-operator-flop-byte-movement-modeling*
*Completed: 2026-05-23*
