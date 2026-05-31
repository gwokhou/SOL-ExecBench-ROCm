---
phase: 84-paper-denominator-accounting-and-claim-boundaries
plan: 01
subsystem: dataset-reporting
tags: [paper-denominator, sidecar, pydantic, markdown, checksums]
requires:
  - phase: 83-closure-contracts-and-provenance-foundation
    provides: execution closure status vocabulary and sidecar provenance patterns
provides:
  - Strict `sol_execbench.paper_denominator_report.v1` report models and builder
  - Deterministic JSON serialization with report checksum
  - Deterministic Markdown renderer and write helpers
  - CPU-safe tests for denominator accounting, bounded refs, and claim boundaries
affects: [phase-84-plan-02, dataset-sidecars, research-credibility-reports]
tech-stack:
  added: []
  patterns: [strict Pydantic sidecar models, stable checksum over normalized payloads, bounded source refs]
key-files:
  created:
    - src/sol_execbench/core/dataset/paper_denominator.py
    - tests/sol_execbench/test_paper_denominator_report.py
  modified: []
key-decisions:
  - "Paper denominator accounting is implemented as a sidecar-only report with no canonical schema changes."
  - "Missing timing, AMD score, AMD SOL, and SOLAR evidence remains evidence_missing/deferred accounting rather than validation authority."
patterns-established:
  - "Use strict `extra='forbid'` models and stable sorted serialization for paper denominator reports."
  - "Render claim-boundary falsehoods as visible Markdown text and machine-readable booleans."
requirements-completed: [DENOM-01, DENOM-02, DENOM-03, DENOM-04, DENOM-05]
duration: 6min
completed: 2026-05-31
---

# Phase 84 Plan 01: Paper Denominator Core Summary

**Strict paper denominator sidecar with deterministic JSON, bounded source refs, checksum, and Markdown claim-boundary wording**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-31T08:23:52Z
- **Completed:** 2026-05-31T08:29:44Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Added `paper_denominator_report.v1` strict Pydantic models, builder, checksum, JSON serialization, Markdown renderer, JSON loader, and write helpers.
- Aggregated readiness, closure, evidence-gap, reason-code, category, problem, workload, and source-ref accounting without embedding source sidecar payloads.
- Preserved explicit false claim boundaries for paper parity, upstream SOLAR parity, leaderboard authority, native-host validation, new hardware validation, full 235-problem validation, and score authority.

## Task Commits

1. **Task 1: Specify denominator report contract behavior** - `86da6c0` (test)
2. **Task 2: Implement strict JSON sidecar models and builder** - `96f42b0` (feat)
3. **Task 3: Add deterministic Markdown and write helpers** - `09a5d05` (feat)

## Files Created/Modified

- `src/sol_execbench/core/dataset/paper_denominator.py` - Strict report models, deterministic builder, bounded source refs, checksum, Markdown renderer, and write helpers.
- `tests/sol_execbench/test_paper_denominator_report.py` - CPU-safe tests for DENOM-01 through DENOM-05 and strict model rejection.

## Decisions Made

- Sidecar-only semantics remain the boundary: no canonical Definition, Workload, Trace, score, timing, evaluator, or primary CLI contract changes were made.
- Evidence gaps produce denominator buckets and next-evidence hints only; they never upgrade paper, SOLAR, leaderboard, score, native-host, or new-hardware authority.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_paper_denominator_report.py -q` - 6 passed

## Known Stubs

None.

## Next Phase Readiness

Plan 84-02 can wire the core helpers into a thin script wrapper, dataset exports, and public contract guardrails.

## Self-Check: PASSED

- Found `src/sol_execbench/core/dataset/paper_denominator.py`
- Found `tests/sol_execbench/test_paper_denominator_report.py`
- Found commits `86da6c0`, `96f42b0`, and `09a5d05`

---
*Phase: 84-paper-denominator-accounting-and-claim-boundaries*
*Completed: 2026-05-31*
