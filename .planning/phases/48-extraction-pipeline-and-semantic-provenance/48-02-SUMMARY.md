---
phase: 48-extraction-pipeline-and-semantic-provenance
plan: 2
subsystem: scoring
tags: [solar, derivation, provenance, bound-graph, estimates]
requires:
  - phase: 48-01
    provides: Strict internal SOLAR derivation evidence dataclasses and parser
provides:
  - SOLAR derivation evidence builder over Definition and Workload
  - Lower-level derivation helper over BoundGraph and OperatorWorkEstimate records
  - Tensor shape, dtype, semantic-axis, source, graph-warning, and estimate-warning provenance tests
affects: [phase-49-high-confidence-family-modeling, phase-50-degraded-complex-family-modeling, phase-51-sidecar-coverage-and-score-guards]
tech-stack:
  added: []
  patterns: [builder over existing bound graph pipeline, immutable sidecar provenance records, TDD red-green task commits]
key-files:
  created:
    - .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-02-SUMMARY.md
  modified:
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_solar_derivation_evidence.py
key-decisions:
  - "Build SOLAR derivation evidence only from Definition, Workload, BoundGraph, and OperatorWorkEstimate inputs."
  - "Keep candidate solution execution explicitly outside the evidence builder boundary."
  - "Map existing graph and estimate warnings into stable graph_warning and estimate_warning provenance strings."
patterns-established:
  - "High-level SOLAR evidence builders call build_bound_graph(definition, workload), estimate_bound_work(graph), then delegate to a prebuilt-graph derivation helper."
  - "Tensor provenance derives semantic axes from declared tensor specs first, then from visible workload/constant axis values when shapes match."
requirements-completed: [DERIVE-07, MODEL-03]
duration: 4min
completed: 2026-05-23
---

# Phase 48 Plan 2: Extraction Plumbing Summary

**SOLAR derivation evidence builder populated from internal bound graph and operator estimate provenance without candidate solution execution**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-23T05:15:04Z
- **Completed:** 2026-05-23T05:18:21Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added focused TDD tests for builder boundaries, tensor provenance, estimate provenance, and candidate solution non-execution.
- Implemented `build_solar_derivation_evidence(definition, workload)` over `build_bound_graph(definition, workload)` and `estimate_bound_work(graph)`.
- Implemented `derive_solar_derivation_evidence(definition, workload, graph, estimates)` to convert existing tensors, graph nodes, graph warnings, and operator estimates into parseable SOLAR sidecar evidence.

## Task Commits

Each task was committed atomically:

1. **Task 48-02-01: Add builder boundary and provenance tests** - `3ed939c` (test)
2. **Task 48-02-02: Implement graph and estimate provenance builder** - `ca2d274` (feat)

## Files Created/Modified

- `src/sol_execbench/core/scoring/solar_derivation.py` - Added high-level and lower-level SOLAR derivation evidence builders plus tensor, group, source, warning, and semantic-axis conversion helpers.
- `tests/sol_execbench/test_solar_derivation_evidence.py` - Added builder boundary tests proving candidate solution code is not accepted or executed and tensor/estimate provenance is recorded.
- `.planning/phases/48-extraction-pipeline-and-semantic-provenance/48-02-SUMMARY.md` - Execution summary and self-check record.

## Decisions Made

- Kept the builder API limited to canonical `Definition` and `Workload`, with no `Solution`, solution path, candidate, submitted-code, subprocess, driver, or execution helper dependency.
- Used prebuilt graph and estimate inputs as the lower-level derivation contract so later phases can test family-specific edge cases without retracing.
- Preserved the existing `BoundGraph` and `OperatorWorkEstimate` dataclasses unchanged and copied their evidence into immutable SOLAR sidecar dataclasses.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` - expected RED failure before implementation: missing `build_solar_derivation_evidence` import.
- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` - 10 passed.
- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` - 42 passed.

## Known Stubs

None.

## Threat Flags

None - the new builder boundary and provenance transformations were already covered by the plan threat model.

## Next Phase Readiness

Phase 48 Plan 3 can add deterministic confidence rules on top of parseable evidence that already records source boundaries, tensor metadata, graph warnings, and estimate warnings.

## Self-Check: PASSED

- Created files exist: `.planning/phases/48-extraction-pipeline-and-semantic-provenance/48-02-SUMMARY.md`.
- Modified files exist: `src/sol_execbench/core/scoring/solar_derivation.py`, `tests/sol_execbench/test_solar_derivation_evidence.py`.
- Task commits are reachable: `3ed939c` and `ca2d274`.
- Focused and phase-gate verification commands passed after implementation.

---
*Phase: 48-extraction-pipeline-and-semantic-provenance*
*Completed: 2026-05-23*
