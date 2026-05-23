---
phase: 51-sidecar-coverage-and-score-guards
plan: 01
subsystem: scoring
tags: [solar, sidecar, coverage, parser, scoring]
requires:
  - phase: 48-extraction-pipeline-and-semantic-provenance
    provides: internal SOLAR derivation evidence sidecars
  - phase: 49-high-confidence-family-modeling
    provides: supported family semantic groups and evidence strictness
  - phase: 50-degraded-complex-family-modeling
    provides: degraded and unsupported semantic group statuses
provides:
  - Internal SOLAR coverage_summary sidecar fields derived from semantic groups
  - Internal SOLAR aggregate_status contract for scored, degraded, and unscored evidence
  - Strict parser support for Phase 51 fields plus Phase 48-50 legacy normalization
affects: [phase-51, phase-52, solar-derivation, score-guards]
tech-stack:
  added: []
  patterns:
    - frozen dataclass sidecar records with deterministic to_dict serialization
    - exact-key parser helpers for nested internal sidecar payloads
key-files:
  created:
    - .planning/phases/51-sidecar-coverage-and-score-guards/51-01-SUMMARY.md
  modified:
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_solar_derivation_evidence.py
key-decisions:
  - "Compute coverage and aggregate status from existing SolarDerivationEvidence.groups and warnings only."
  - "Keep Phase 48-50 payloads parseable by accepting the legacy exact top-level key set and recomputing new fields."
  - "Keep parser strict for unknown top-level and nested Phase 51 coverage fields."
patterns-established:
  - "Coverage sidecar records use deterministic count maps, sorted IDs, and group/node-tied provenance refs."
  - "Aggregate status applies unscored > degraded > scored precedence and sets score_eligible only for scored evidence."
requirements-completed: [REPORT-01, REPORT-02, TEST-03]
duration: 5min
completed: 2026-05-23
---

# Phase 51 Plan 01: Sidecar Coverage And Score Guards Summary

**Internal SOLAR sidecars now expose deterministic coverage summaries and aggregate scored/degraded/unscored status without changing public contracts**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-23T08:58:48Z
- **Completed:** 2026-05-23T09:03:52Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added frozen coverage/aggregate dataclasses and deterministic serialization for `coverage_summary` and `aggregate_status`.
- Derived family/status counts, missing and unsupported patterns, degraded/unsupported/estimated node IDs, and provenance refs from existing semantic groups.
- Updated `solar_derivation_from_dict()` to accept strict Phase 51 payloads and exact Phase 48-50 legacy payloads while preserving unknown-field rejection.
- Added round-trip, legacy normalization, empty-group unscored, aggregate precedence, nested unknown, and deterministic coverage tests.

## Task Commits

1. **TDD RED: coverage sidecar tests** - `da1c01c` (`#51 - Add failing SOLAR coverage sidecar tests`)
2. **Tasks 51-01-01 through 51-01-03: coverage records, aggregation, parser normalization** - `88d3322` (`#51 - Derive SOLAR coverage aggregate sidecars`)

## Files Created/Modified

- `src/sol_execbench/core/scoring/solar_derivation.py` - Adds internal coverage/aggregate records, group-derived aggregation helpers, exact nested parsers, and legacy top-level payload support.
- `tests/sol_execbench/test_solar_derivation_evidence.py` - Adds TEST-03 coverage for new machine-verifiable sidecar fields, strict parsing, and deterministic aggregation.
- `.planning/phases/51-sidecar-coverage-and-score-guards/51-01-SUMMARY.md` - Execution summary and verification record.

## Decisions Made

- Coverage is computed lazily during `SolarDerivationEvidence.to_dict()` from internal groups and warnings, avoiding a second extractor or public schema change.
- Parser validation accepts exactly the old Phase 48-50 top-level key set or the new Phase 51 key set; any mixed or unknown key set fails.
- Degraded aggregate evidence is parseable but not `score_eligible`; only fully scored aggregate evidence is score eligible.

## Deviations from Plan

None - plan executed within the requested 51-01 scope.

## Known Stubs

None.

## Threat Flags

None. The new serialized sidecar parser surface was already covered by the plan threat model and uses exact-key validation.

## Issues Encountered

- TDD RED failed as expected because `coverage_summary` was absent before implementation.
- A first implementation pass missed the integer parser helper for `group_count`; fixed before the GREEN commit.

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "coverage or aggregate or parser or unknown or malformed or round_trip or deterministic" -n 0 -x` - 35 passed, 15 deselected.
- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_family_modeling.py -n 0 -x` - 64 passed.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_solar_derivation_evidence.py` - passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 51-02 can build on `aggregate_status` and `coverage_summary` without changing public problem schemas, CLI output, canonical trace JSONL, or score report surfaces.

## Self-Check: PASSED

- Found `.planning/phases/51-sidecar-coverage-and-score-guards/51-01-SUMMARY.md`.
- Found task commit `da1c01c`.
- Found task commit `88d3322`.

---
*Phase: 51-sidecar-coverage-and-score-guards*
*Completed: 2026-05-23*
