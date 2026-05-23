---
phase: 48-extraction-pipeline-and-semantic-provenance
plan: 1
subsystem: scoring
tags: [solar, derivation, provenance, sidecar, parser, dataclasses]
requires:
  - phase: 47-derivation-contract-and-golden-fixture-matrix
    provides: SOLAR fixture confidence/status vocabulary and public boundary guardrails
provides:
  - Strict internal SOLAR derivation evidence dataclasses
  - JSON-safe sidecar parser and serializer for semantic provenance
  - Focused parser, serializer, malformed payload, and sidecar boundary tests
affects: [phase-49-high-confidence-family-modeling, phase-50-degraded-complex-family-modeling, phase-51-sidecar-coverage-and-score-guards]
tech-stack:
  added: []
  patterns: [frozen dataclass sidecars, strict nested parser, JSON-safe to_dict round trips]
key-files:
  created:
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_solar_derivation_evidence.py
  modified: []
key-decisions:
  - "Keep SOLAR derivation evidence internal and sidecar-only with explicit source_boundary booleans."
  - "Reuse the existing EstimateConfidence vocabulary while serializing confidence as JSON-safe strings."
patterns-established:
  - "SOLAR evidence records use frozen dataclasses with tuple internals and list-based to_dict output."
  - "Nested sidecar parsers require every field and raise deterministic ValueError messages for malformed payloads."
requirements-completed: [DERIVE-07, MODEL-03]
duration: 4min
completed: 2026-05-23
---

# Phase 48 Plan 1: Evidence Contract Parser Summary

**Internal SOLAR derivation evidence sidecar with strict parser/serializer coverage for semantic provenance and public-boundary protection**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-23T05:07:50Z
- **Completed:** 2026-05-23T05:11:27Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `SolarDerivationEvidence`, semantic group, subrole, tensor, and source dataclasses for sidecar-only SOLAR evidence.
- Implemented `solar_derivation_from_dict()` with required-field checks, schema validation, confidence/status validation, non-empty ID/name checks, JSON-safe list parsing, and source-boundary boolean validation.
- Added focused tests proving round trips preserve provenance, malformed nested payloads fail deterministically, and canonical trace/public schema/candidate execution boundaries remain false in the sidecar payload.

## Task Commits

Each task was committed atomically:

1. **Task 48-01-01: Add evidence contract parser tests** - `2ed5ace` (test)
2. **Task 48-01-02: Implement strict internal evidence model** - `40b8d79` (feat)

## Files Created/Modified

- `src/sol_execbench/core/scoring/solar_derivation.py` - Internal frozen evidence dataclasses plus strict parser helpers for SOLAR derivation sidecars.
- `tests/sol_execbench/test_solar_derivation_evidence.py` - Focused contract tests for JSON-safe round trips, malformed payload rejection, invalid confidence/status rejection, and source-boundary preservation.

## Decisions Made

- Kept the new evidence module out of public Pydantic models, canonical trace JSONL, CLI, driver, solution, and candidate execution paths.
- Stored confidence internally as existing `EstimateConfidence` values when parsed, while accepting dataclass construction with valid confidence strings and emitting JSON-safe strings.
- Preserved `source_boundary` booleans exactly instead of coercing non-boolean values.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` - 7 passed.
- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` - 39 passed.

## Known Stubs

None.

## Threat Flags

None - the new sidecar payload parser and public-boundary protections were already covered by the plan threat model.

## Next Phase Readiness

Phase 48 Plan 2 can consume the internal evidence contract for extractor plumbing without changing public benchmark artifacts or primary CLI behavior.

## Self-Check: PASSED

- Created files exist: `src/sol_execbench/core/scoring/solar_derivation.py`, `tests/sol_execbench/test_solar_derivation_evidence.py`, and this summary.
- Task commits are reachable: `2ed5ace` and `40b8d79`.
- Focused and phase-gate verification commands passed after implementation.

---
*Phase: 48-extraction-pipeline-and-semantic-provenance*
*Completed: 2026-05-23*
