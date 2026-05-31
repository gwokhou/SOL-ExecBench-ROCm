---
phase: 87-amd-sol-solar-bound-sanity-evidence
plan: 01
subsystem: scoring
tags: [amd-bound-sanity, amd-sol, solar, sidecar, reporting, tdd]
requires:
  - phase: 84-paper-denominator-accounting-and-claim-boundaries
    provides: deterministic sidecar report patterns and claim-boundary wording
  - phase: 85-compatibility-matrix-schema-export-and-semantic-diff
    provides: Compatibility Matrix evidence surface used as bounded input
  - phase: 86-dataset-runner-hardening-integration
    provides: execution closure sidecar evidence consumed by this report
provides:
  - strict amd_bound_sanity.v1 diagnostic report contract
  - deterministic JSON checksum and Markdown rendering helpers
  - CPU-safe tests for SANITY-01 through SANITY-04
affects: [phase-88-docs, reporting, amd-sol, solar-derivation, public-contract-guardrails]
tech-stack:
  added: []
  patterns: [strict Pydantic sidecar models, stable_json_checksum, sorted JSON, deterministic Markdown]
key-files:
  created:
    - src/sol_execbench/core/scoring/amd_bound_sanity.py
    - tests/sol_execbench/test_amd_bound_sanity.py
  modified: []
key-decisions:
  - "amd_bound_sanity.v1 remains a scoring sidecar/reporting artifact and does not modify score eligibility or canonical schemas."
  - "Diagnostic totals count primary status plus provisional risk flags so degraded/provisional rows remain visible without changing source score support."
  - "Claim boundaries are literal false fields plus visible Markdown wording; provisional RDNA 4 risk is a risk flag, not validation."
patterns-established:
  - "Bounded source refs carry path/ref/schema_version/checksum only; full source payloads are not embedded in report output."
  - "Existing evidence dictionaries are merged by workload UUID without probing filesystem, Docker, ROCm, GPU APIs, or package managers."
requirements-completed: [SANITY-01, SANITY-02, SANITY-03, SANITY-04]
duration: 18min
completed: 2026-05-31
---

# Phase 87 Plan 01: AMD Bound Sanity Core Summary

**Strict amd_bound_sanity.v1 sidecar models with existing-evidence diagnostic rollups, bounded refs, false authority boundaries, checksum, and deterministic Markdown**

## Performance

- **Duration:** 18 min
- **Started:** 2026-05-31T10:58:45Z
- **Completed:** 2026-05-31T11:16:34Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Added `amd_bound_sanity.v1` strict Pydantic report models and builder over inline/existing evidence dictionaries.
- Added diagnostic status rollups for `scored`, `degraded`, `unscored`, `unsupported`, `provisional`, and `missing_evidence` without changing AMD-native score `supported` semantics.
- Added deterministic sorted JSON, stable checksum, Markdown renderer, and JSON/Markdown write helpers with explicit negative claim boundaries.

## Task Commits

1. **Task 1: Specify bound sanity report contract behavior** - `0b3ae91` (test RED)
2. **Task 2: Implement strict JSON models and report builder** - `fd38dc0` (feat GREEN)
3. **Task 3: Add deterministic Markdown renderer and write helpers** - `fd38dc0` (feat GREEN, same core module commit)

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_bound_sanity.py` - Strict report models, source refs, status rollups, evidence gaps, checksum, Markdown renderer, and write helpers.
- `tests/sol_execbench/test_amd_bound_sanity.py` - CPU-safe TDD contract tests for SANITY-01 through SANITY-04.

## Decisions Made

- Diagnostic status is derived into a separate sanity layer while source statuses and AMD score support remain visible as source evidence.
- Provisional RDNA 4 model risk is represented as `provisional_rdna4_model_risk=true`, not as AMD SOL/SOLAR model validation.
- Markdown includes visible negative wording for upstream SOLAR equivalence, model validation, paper parity, leaderboard authority, score authority upgrade, CDNA 3, MI300X, CDNA 4, native-host validation, and new-hardware validation.

## Deviations from Plan

None - plan executed within the requested sidecar/reporting scope.

## Issues Encountered

- The first GREEN run exposed a merge bug where closure, AMD SOL, SOLAR, and score evidence for the same workload UUID overwrote each other. Fixed before committing the implementation.

## Known Stubs

None. Stub scan only found intentional empty/`None` fixture inputs in missing-evidence tests.

## Threat Flags

None. The new surface is the planned local sidecar dictionary-to-report boundary and Markdown rendering boundary covered by the plan threat model.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_bound_sanity.py -q` failed as expected during RED with missing module.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_bound_sanity.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_solar_derivation_evidence.py -q` passed: 117 passed.

## Self-Check: PASSED

- Created files exist: `src/sol_execbench/core/scoring/amd_bound_sanity.py`, `tests/sol_execbench/test_amd_bound_sanity.py`.
- Commits exist: `0b3ae91`, `fd38dc0`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 87-02 can add the thin script wrapper and public contract guardrails on top of the core report helpers.

---
*Phase: 87-amd-sol-solar-bound-sanity-evidence*
*Completed: 2026-05-31*
