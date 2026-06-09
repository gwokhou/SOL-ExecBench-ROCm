---
phase: 171-custom-input-coverage-recompute
plan: "01"
subsystem: coverage
tags: [custom-inputs, transition-ledger, readiness, coverage, rocm]

# Dependency graph
requires:
  - phase: 170
    provides: Custom input evaluator readiness with entrypoint support and reclassification
provides:
  - Transition ledger covering all 55 original custom-input readiness blockers
  - Updated coverage artifacts with revised blocker counts
  - CPU-safe regression tests for ledger completeness and classification fidelity
affects: [172, 173, 174, coverage-recompute]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Before/after transition ledger with baseline checksum and denominator assertion"
    - "Transition classification mapping readiness status pairs to named transitions"
    - "Residual blocker class selection from a required code set"

key-files:
  created:
    - scripts/build_custom_input_transition_ledger.py
    - tests/sol_execbench/test_custom_input_transition_ledger.py
  modified: []

key-decisions:
  - "All 55 custom_input_blocked problems promoted to ready after Phase 170 entrypoint support"
  - "Transition ledger records baseline checksum and path per D-01 and D-02"
  - "Coverage refresh uses profiler-upgrade-260609 timing dir first for correct profiler_backed counts"

patterns-established:
  - "Transition ledger: fixed baseline -> fresh readiness recompute -> per-problem before/after record"
  - "Residual class selection from required set, never generic strings"

requirements-completed: [COV-01, COV-02]

# Metrics
duration: pre-committed
completed: 2026-06-09
---

# Phase 171: Custom Input Coverage Recompute Summary

**Transition ledger confirming all 55 custom-input blockers promoted to ready, updated coverage showing readiness_blocked reduced from 114 to 59, and 235-problem denominator stable**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-06-09
- **Completed:** 2026-06-09
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Transition ledger covering all 55 original custom_input_blocked problems with explicit before/after dispositions
- All 55 problems promoted from custom_input_blocked to ready (100% promotion rate)
- Updated coverage: readiness_blocked reduced from 114 to 59, zero custom_input_blocked remaining
- 235-problem denominator confirmed stable across recomputation
- CPU-safe test suite (23 tests) covering cohort extraction, transition classification, residual class validation, and denominator stability

## Task Commits

Each task was committed atomically:

1. **Task 1: Build transition ledger with fixed baseline and fresh readiness recompute** - `8535a71` (feat)
2. **Task 2: Execute transition ledger, update coverage artifacts** - (local artifacts only, out/ is gitignored)

## Files Created/Modified
- `scripts/build_custom_input_transition_ledger.py` - Transition ledger generator with baseline loading, fresh readiness classification, and per-problem transition records
- `tests/sol_execbench/test_custom_input_transition_ledger.py` - 23 CPU-safe tests for cohort extraction, transition classification, workload unavailability, residual class validation, and denominator mismatch

## Key Results

| Metric | Before | After |
|--------|--------|-------|
| custom_input_blocked | 55 | 0 |
| readiness_blocked | 114 | 59 |
| ready (missing profiler timing) | 0 | 55 |
| profiler_backed | 62 | 59 |
| timing_fallback | 42 | 48 |
| reference_oom_blocked | 15 | 14 |
| problem_denominator | 235 | 235 |

## Decisions Made
- Coverage refresh uses `rdna4-profiler-upgrade-260609` timing directory first to preserve profiler_backed counts; the default `rdna4-timing-evidence/timing` directory has device_events (non-profiler) sidecars that shadow profiler-backed ones when listed first
- No real RDNA4 smoke execution attempted (execution_environment_unavailable per D-08); readiness recompute satisfies the hard gate per D-07

## Deviations from Plan

None - plan executed exactly as specified.

## Issues Encountered

- Initial coverage refresh showed 0 profiler_backed because timing evidence directories were ordered with the device_events fallback dir first; resolved by placing the profiler-backed timing dir first in the evidence search order

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 55 custom-input blockers have explicit dispositions, ready for Phase 172 (Quant readiness) and Phase 173 (FlashInfer readiness)
- Transition ledger at `out/rdna4-custom-input-transition-ledger/transition-ledger.json` available for Phase 174 final closure
- Updated coverage artifacts at `out/rdna4-coverage-current/` reflect current readiness state

---
*Phase: 171-custom-input-coverage-recompute*
*Completed: 2026-06-09*
