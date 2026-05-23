---
phase: 48-extraction-pipeline-and-semantic-provenance
plan: 4
subsystem: scoring
tags: [solar, derivation, fixture-contract, public-contract, amd-score]
requires:
  - phase: 48-03
    provides: Deterministic SOLAR semantic groups and conservative confidence/status classification
  - phase: 47
    provides: Golden SOLAR derivation fixture matrix and public contract guardrails
provides:
  - Fixture-driven Phase 47 expectation round-trip coverage through Phase 48 evidence
  - Public-schema, trace JSONL, and primary CLI exclusions for Phase 48 evidence fields and options
  - AMD-native score eligibility regression guard covering SOLAR sidecar module imports
affects: [phase-49-high-confidence-family-modeling, phase-50-degraded-complex-family-modeling, phase-51-sidecar-coverage-and-score-guards]
tech-stack:
  added: []
  patterns: [fixture-to-sidecar evidence round-trip, sidecar-only public contract guardrail]
key-files:
  created:
    - .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-04-SUMMARY.md
  modified:
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_solar_derivation_evidence.py
    - tests/sol_execbench/test_public_contract_guardrails.py
key-decisions:
  - "Keep Phase 47 fixture expectations as data-only inputs and round-trip them through Phase 48 evidence without executing fixture references."
  - "Add `required_evidence` to the internal SOLAR semantic group sidecar because fixture expectations require exact preservation of present evidence as well as missing evidence."
  - "Keep Phase 48 evidence names and CLI switches excluded from canonical public schemas, trace JSONL, and primary `sol-execbench --help`."
patterns-established:
  - "Fixture expectations can be represented as parseable `SolarDerivationEvidence` payloads with exact family, subrole, confidence, status, required evidence, missing evidence, warning, and boundary preservation."
  - "Public guardrails assert both top-level field absence and string absence for Phase 48 schema/version markers in canonical model dumps."
requirements-completed: [DERIVE-07, MODEL-03, MODEL-04]
duration: 4min
completed: 2026-05-23
---

# Phase 48 Plan 4: Fixture Coverage And Public Contract Guardrails Summary

**Phase 47 fixture expectations now round-trip through internal Phase 48 SOLAR evidence while public schemas, CLI help, trace JSONL, and AMD-native score eligibility stay unchanged**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-23T05:29:18Z
- **Completed:** 2026-05-23T05:33:24Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added fixture-driven coverage proving all Phase 47 SOLAR families and fixture classes are representable as parseable Phase 48 evidence payloads.
- Preserved positive, degraded, and unsupported fixture expectations exactly, including subroles, confidence, status, required evidence, missing evidence, warnings, and non-claim boundaries.
- Expanded public contract guardrails for Phase 48 evidence field names, schema version strings, and primary CLI option exclusions.
- Added an AMD-native score eligibility guard proving importing the internal SOLAR sidecar module does not alter v1/v2 score artifacts or evidence references.

## Task Commits

Each task was committed atomically:

1. **Task 48-04-01: Add fixture-driven evidence coverage** - `a340858` (test)
2. **Task 48-04-02: Extend public contract and CLI guardrails** - `814727f` (test)

## Files Created/Modified

- `src/sol_execbench/core/scoring/solar_derivation.py` - Added internal `required_evidence` round-trip support on semantic group evidence and deterministic required-evidence derivation from tensor/estimate provenance.
- `tests/sol_execbench/test_solar_derivation_evidence.py` - Added Phase 47 fixture-to-Phase 48 evidence round-trip coverage and explicit scope-boundary assertions.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Added Phase 48 forbidden field/option assertions and AMD-native score eligibility guard around SOLAR sidecar imports.
- `.planning/phases/48-extraction-pipeline-and-semantic-provenance/48-04-SUMMARY.md` - Execution summary and verification record.

## Decisions Made

- The internal SOLAR sidecar now carries `required_evidence` on each semantic group so fixture expectations can be preserved exactly; this remains internal and is guarded from canonical public models.
- Fixture references remain data only. Tests load fixture JSON and construct evidence payloads from expectation fields without invoking fixture reference text.
- Primary CLI defaults and AMD-native score eligibility stay unchanged; Phase 48 remains a sidecar evidence foundation for later phases.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added required evidence to the internal sidecar**
- **Found during:** Task 48-04-01
- **Issue:** The plan required fixture `required_evidence` to be recorded and preserved exactly, but the existing `SolarSemanticGroupEvidence` schema had no field for present evidence.
- **Fix:** Added internal `required_evidence` serialization/parsing and deterministic derivation for generated groups.
- **Files modified:** `src/sol_execbench/core/scoring/solar_derivation.py`, `tests/sol_execbench/test_solar_derivation_evidence.py`
- **Verification:** `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x` passed.
- **Committed in:** `a340858`

---

**Total deviations:** 1 auto-fixed (1 missing critical functionality)
**Impact on plan:** The deviation was necessary for exact fixture-contract preservation and remained internal to Phase 48 sidecar evidence. No public schema, trace JSONL, primary CLI, or score eligibility behavior changed.

## Issues Encountered

- Initial fixture evidence test import used a package-style `tests.sol_execbench...` path, but this repository loads nearby fixture helpers through a local test directory path. The import was aligned with `test_solar_derivation_contract.py`.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x` - 33 passed.
- `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -n 0 -x` - 17 passed.
- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` - 50 passed.
- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_v2.py -n 0` - 19 passed.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py` - passed.

## Known Stubs

None. Empty tensors in fixture evidence payloads are intentional because the tests validate Phase 47 expectation representability without executing fixture references.

## Threat Flags

None - fixture handling, public field/option exclusion, and AMD score eligibility were covered by the plan threat model.

## Next Phase Readiness

Phase 48 is ready for verification and later Phase 49/50 family modeling. The internal evidence contract can now represent fixture expectations exactly while preserving sidecar-only boundaries.

## Self-Check: PASSED

- Created file exists: `.planning/phases/48-extraction-pipeline-and-semantic-provenance/48-04-SUMMARY.md`.
- Modified files exist: `src/sol_execbench/core/scoring/solar_derivation.py`, `tests/sol_execbench/test_solar_derivation_evidence.py`, and `tests/sol_execbench/test_public_contract_guardrails.py`.
- Task commits are reachable: `a340858` and `814727f`.
- Focused, phase-gate, regression, and Ruff verification commands passed after implementation.

---
*Phase: 48-extraction-pipeline-and-semantic-provenance*
*Completed: 2026-05-23*
