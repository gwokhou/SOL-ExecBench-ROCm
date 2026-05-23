---
phase: 48-extraction-pipeline-and-semantic-provenance
plan: 3
subsystem: scoring
tags: [solar, derivation, semantic-groups, confidence, provenance]
requires:
  - phase: 48-02
    provides: SOLAR derivation evidence builder over bound graph and estimate provenance
provides:
  - Deterministic SOLAR semantic grouping by visible operation family
  - Pure conservative confidence/status classifier for supported, inexact, and unsupported evidence
  - Stable missing evidence, warning prefix, rationale, group, and subrole ordering
affects: [phase-49-high-confidence-family-modeling, phase-50-degraded-complex-family-modeling, phase-51-sidecar-coverage-and-score-guards]
tech-stack:
  added: []
  patterns: [pure confidence classifier, deterministic semantic grouping, conservative sidecar evidence]
key-files:
  created:
    - .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-03-SUMMARY.md
  modified:
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_solar_derivation_evidence.py
key-decisions:
  - "Keep `classify_solar_confidence` pure and limited to already-built graph, tensor, estimate, and subrole evidence."
  - "Map supported, inexact, and unsupported confidence to scored, degraded, and unscored status conservatively."
  - "Use graph semantics for subrole provenance while retaining estimate provenance at the semantic-group level."
patterns-established:
  - "Semantic groups are sorted by first node ID and assigned stable `group:{family}:{index}` identifiers."
  - "Degraded and unsupported groups carry sorted missing evidence, sorted warnings, and explicit rationale text."
requirements-completed: [DERIVE-07, MODEL-03, MODEL-04]
duration: 5min
completed: 2026-05-23
---

# Phase 48 Plan 3: Semantic Grouping And Confidence Summary

**Deterministic SOLAR semantic groups with conservative supported, degraded, and unscored confidence classification**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-23T05:21:55Z
- **Completed:** 2026-05-23T05:26:11Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added RED tests for complete supported/scored, incomplete inexact/degraded, and unsupported/unscored semantic evidence.
- Implemented `classify_solar_confidence` as a pure helper with no file, hardware, candidate, or benchmark-runtime dependency.
- Replaced one-estimate/one-group emission with deterministic family grouping, sorted subroles, sorted missing evidence, and stable warning prefixes.
- Preserved sidecar-only behavior and public contract guardrails.

## Task Commits

Each task was committed atomically:

1. **Task 48-03-01: Add semantic grouping and confidence tests** - `3860b81` (test)
2. **Task 48-03-02: Implement semantic grouping and confidence rules** - `3ab313d` (feat)

## Files Created/Modified

- `src/sol_execbench/core/scoring/solar_derivation.py` - Added `SolarConfidenceClassification`, pure classification, deterministic semantic grouping helpers, stable aggregate warnings, and conservative status mapping.
- `tests/sol_execbench/test_solar_derivation_evidence.py` - Added confidence/status and deterministic serialization coverage for supported, degraded, and unsupported groups.
- `.planning/phases/48-extraction-pipeline-and-semantic-provenance/48-03-SUMMARY.md` - Execution summary and self-check record.

## Decisions Made

- `classify_solar_confidence` receives only already-derived in-memory evidence and does not inspect files, invoke hardware, execute candidates, or inspect benchmark runtime results.
- Supported groups require recognized family evidence, core subroles, tensor shape/dtype/source/semantic-axis evidence, formula inputs, byte evidence, and axis provenance.
- Incomplete visible evidence becomes `inexact`/`degraded`; unsupported or subrole-less evidence becomes `unsupported`/`unscored`.
- Subrole source provenance now comes from graph semantics (`ast` or `fx`), while group-level source provenance remains estimate-backed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated stale subrole source assertion**
- **Found during:** Task 48-03-02
- **Issue:** A pre-existing Phase 48-02 test expected subrole source provenance to be `estimate`, but Plan 48-03 requires subroles to come from existing graph semantics.
- **Fix:** Updated the assertion to expect group source provenance from `estimate` and subrole source provenance from `ast`.
- **Files modified:** `tests/sol_execbench/test_solar_derivation_evidence.py`
- **Verification:** `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x` passed.
- **Committed in:** `3ab313d`

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** The fix aligned prior coverage with the new semantic grouping contract. No scope expansion.

## Issues Encountered

None beyond the auto-fixed stale assertion above.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x` - expected RED failure before implementation: missing `classify_solar_confidence` import.
- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x` - 30 passed.
- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` - 46 passed.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_solar_derivation_evidence.py` - passed.

## Known Stubs

None. Stub scan hits were intentional `None` and empty-dict fixture values used to exercise missing evidence behavior.

## Threat Flags

None - confidence classification, missing evidence generation, warnings, and rationale were covered by the plan threat model.

## Next Phase Readiness

Phase 48 Plan 4 can now validate fixture-driven coverage and public contract guardrails against deterministic semantic group, subrole, status, warning, and missing-evidence output.

## Self-Check: PASSED

- Created file exists: `.planning/phases/48-extraction-pipeline-and-semantic-provenance/48-03-SUMMARY.md`.
- Modified files exist: `src/sol_execbench/core/scoring/solar_derivation.py`, `tests/sol_execbench/test_solar_derivation_evidence.py`.
- Task commits are reachable: `3860b81` and `3ab313d`.
- Focused and phase-gate verification commands passed after implementation.

---
*Phase: 48-extraction-pipeline-and-semantic-provenance*
*Completed: 2026-05-23*
