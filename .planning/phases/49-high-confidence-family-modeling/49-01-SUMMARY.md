---
phase: 49-high-confidence-family-modeling
plan: 01
subsystem: scoring
tags: [solar, evidence, parser, guardrails, amd-sol]

requires:
  - phase: 48-extraction-pipeline-and-semantic-provenance
    provides: Internal SOLAR derivation sidecar, semantic groups, strict parser, and public boundary guardrails
provides:
  - Group-local formula, byte, and bound evidence contracts inside SOLAR semantic groups
  - Strict nested parser validation for malformed formula, byte, and bound evidence payloads
  - Public guardrails keeping Phase 49 sidecar field names out of canonical schemas, primary CLI help, and AMD-native score evidence refs
affects: [49-high-confidence-family-modeling, 51-sidecar-coverage-and-score-guards]

tech-stack:
  added: []
  patterns: [frozen dataclass sidecars, exact-key parser helpers, deterministic node-id serialization]

key-files:
  created:
    - .planning/phases/49-high-confidence-family-modeling/49-01-SUMMARY.md
  modified:
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_solar_derivation_evidence.py
    - tests/sol_execbench/test_public_contract_guardrails.py

key-decisions:
  - "Nested formula, byte, and bound evidence stays inside SolarSemanticGroupEvidence as internal sidecar data."
  - "Bound evidence reuses AMD SOL v2 compute/memory/SOL semantics with an internal default gfx1200 hardware model reference."
  - "Public guardrails check new sidecar field names at canonical and score-evidence boundaries without forbidding existing AMD SOL artifact bound fields."

patterns-established:
  - "Sidecar evidence records serialize deterministically by node_id and parse through exact-key nested helpers."
  - "required_evidence references formula_evidence:{node_id}, byte_evidence:{node_id}, and bound_evidence:{node_id} only when records exist."

requirements-completed: [MODEL-01, MODEL-02, MODEL-05]

duration: 6min
completed: 2026-05-23
---

# Phase 49 Plan 01: Shared Group-Local Evidence Summary

**Internal SOLAR semantic groups now carry strict formula, byte, and AMD SOL-style bound evidence records.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-23T06:13:07Z
- **Completed:** 2026-05-23T06:19:06Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added frozen `SolarFormulaEvidence`, `SolarByteEvidence`, and `SolarBoundEvidence` records and nested them in `SolarSemanticGroupEvidence`.
- Populated formula and byte evidence from `OperatorWorkEstimate`; populated bound evidence with AMD SOL v2 compute/memory/SOL math using the internal default `gfx1200` hardware model reference.
- Extended strict parsing to reject unknown fields, missing fields, invalid confidence values, invalid limiting resources, non-numeric values, non-finite values, and negative byte/bound values.
- Extended guardrails so new Phase 49 field names stay out of canonical `Definition`, `Workload`, `Trace`, primary CLI help, and AMD-native score evidence refs while preserving existing AMD SOL bound artifact fields.

## Task Commits

1. **Tasks 49-01-01 through 49-01-03 RED tests:** `7602c8b` (`#49 - Add sidecar evidence contract tests`)
2. **Tasks 49-01-01 and 49-01-02 implementation:** `dea3101` (`#49 - Add group-local SOLAR evidence contracts`)

## Files Created/Modified

- `src/sol_execbench/core/scoring/solar_derivation.py` - Adds group-local evidence dataclasses, serializer population, AMD SOL-style bound helper, and strict nested parsers.
- `tests/sol_execbench/test_solar_derivation_evidence.py` - Adds round-trip, malformed payload, deterministic ordering, and required evidence reference coverage.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Adds sidecar-only public boundary guardrails for new evidence field names.
- `.planning/phases/49-high-confidence-family-modeling/49-01-SUMMARY.md` - Records execution outcome.

## Decisions Made

- Kept new evidence fields internal to the SOLAR derivation sidecar and nested per semantic group per D-49-06.
- Used the built-in `default_amd_hardware_models()["gfx1200"]` only inside SOLAR bound evidence derivation, without routing it into AMD-native score eligibility.
- Narrowed public guardrails to assert score evidence refs and canonical surfaces, not existing AMD SOL v1/v2 artifact internals such as `compute_bound_ms`.

## Deviations from Plan

None - plan executed as written.

## Issues Encountered

None.

## Known Stubs

None. Stub scan only found an intentional malformed-test fixture with empty `formula_inputs`.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: sidecar-parser | `src/sol_execbench/core/scoring/solar_derivation.py` | Added nested sidecar parser surface for formula, byte, and bound evidence. Mitigated by exact-key parsing and strict value validation. |

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` - 54 passed.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py` - passed.

## Self-Check: PASSED

- Created summary file exists.
- Task commits `7602c8b` and `dea3101` exist.
- Required files were modified and verification passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plans 49-02 through 49-04 can now populate the shared group-local formula, byte, and bound evidence records for promoted families without changing public schemas or AMD-native scoring eligibility.

---
*Phase: 49-high-confidence-family-modeling*
*Completed: 2026-05-23*
