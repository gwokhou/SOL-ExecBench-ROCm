---
phase: 51-sidecar-coverage-and-score-guards
plan: 02
subsystem: scoring
tags: [amd-native-score, solar-derivation, score-guards, public-contracts]

requires:
  - phase: 51-sidecar-coverage-and-score-guards
    provides: "51-01 SOLAR coverage summary and aggregate status sidecars"
provides:
  - "Optional internal SOLAR aggregate score guard for AMD-native workload scoring"
  - "Trace and suite score builder propagation for workload UUID keyed SOLAR sidecars"
  - "Public score evidence-ref guardrails for Phase 51 internal SOLAR fields"
affects: [phase-51, phase-52, amd-native-score, solar-derivation]

tech-stack:
  added: []
  patterns:
    - "Keyword-only internal score guard inputs with neutral None defaults"
    - "Deterministic warning preservation through existing AMD-native warning strings"

key-files:
  created:
    - ".planning/phases/51-sidecar-coverage-and-score-guards/51-02-SUMMARY.md"
  modified:
    - "src/sol_execbench/core/scoring/amd_score.py"
    - "tests/sol_execbench/test_amd_native_score.py"
    - "tests/sol_execbench/test_public_contract_guardrails.py"

key-decisions:
  - "Explicit SOLAR aggregate status unscored suppresses AMD-native score output."
  - "Degraded SOLAR aggregate status preserves numeric AMD-native scoring when existing numeric inputs are complete."
  - "Internal SOLAR guard inputs do not add public score evidence_refs or change claim_level."

patterns-established:
  - "AMD-native score builders may accept internal sidecars via keyword-only optional arguments."
  - "Missing sidecar map entries are treated as absent evidence, not explicit unscored evidence."

requirements-completed: [REPORT-03]

duration: 6min
completed: 2026-05-23
---

# Phase 51 Plan 02: SOLAR Score Guard Summary

**Internal SOLAR aggregate status now guards AMD-native scores without changing public score evidence refs or claim level**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-23T09:04:00Z
- **Completed:** 2026-05-23T09:10:10Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added optional keyword-only `solar_derivation` input to workload and trace AMD-native score builders.
- Added optional `solar_derivations_by_workload_uuid` suite map support with neutral behavior for missing entries.
- Explicit `unscored` SOLAR aggregate status now returns `score=None` with deterministic unscored warnings.
- Explicit `degraded` SOLAR aggregate status now appends deterministic degraded warnings while preserving numeric scores when measured latency, baseline latency, and SOL bound inputs are complete.
- Added tests proving internal SOLAR fields do not leak into public `evidence_refs` and `claim_level` remains `amd-native-derived`.

## Task Commits

1. **Tasks 51-02-01 through 51-02-03 RED:** `70bf2c7` (`#51 - Add failing SOLAR score guard tests`)
2. **Tasks 51-02-01 through 51-02-03 GREEN:** `8bce00a` (`#51 - Guard AMD scores with SOLAR aggregate status`)

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_score.py` - Optional SOLAR aggregate guard, trace forwarding, and suite UUID map forwarding.
- `tests/sol_execbench/test_amd_native_score.py` - Workload, trace, suite, degraded, unscored, and absent-sidecar neutrality tests.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Public evidence ref and claim-level guardrail tests.
- `.planning/phases/51-sidecar-coverage-and-score-guards/51-02-SUMMARY.md` - Execution summary.

## Decisions Made

- Used `SolarAggregateStatus | SolarDerivationEvidence` as the internal guard input shape so callers can pass either the aggregate produced by Plan 51-01 or the full internal sidecar.
- Kept all guard parameters keyword-only and defaulted to `None` so existing AMD-native score callers remain unchanged.
- Reused the existing AMD-native degraded/unscored warning constants and `_unique()` deduplication path instead of introducing new public warning categories.

## Deviations from Plan

None - plan executed as written.

## Issues Encountered

None.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_native_score.py -k "solar or degraded or unscored or workload_score" -n 0 -x` - 11 passed, 7 deselected.
- `uv run pytest tests/sol_execbench/test_amd_native_score.py -k "trace or suite or solar or degraded or unscored" -n 0 -x` - 14 passed, 4 deselected.
- `uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py -k "solar or score or degraded or unscored or evidence_refs" -n 0 -x` - 26 passed, 12 deselected.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_score.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py` - passed.

## Known Stubs

None.

## Threat Flags

None. The only new trust-boundary behavior is the planned internal SOLAR sidecar to AMD-native score guard path, covered by tests for explicit unscored handling, degraded warnings, absent-sidecar neutrality, and public evidence-ref containment.

## Risks

- The guard remains internal and opt-in; Phase 52 still needs to decide any public runner/reporting surface.
- `SolarDerivationEvidence` aggregate status is derived through its existing `to_dict()` path, so future sidecar schema changes should keep that serialized aggregate shape stable or update this bridge.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 51-03 can build on these score guard hooks and the public boundary assertions without changing canonical trace JSONL, CLI behavior, or public problem schemas.

## Self-Check: PASSED

- Found `.planning/phases/51-sidecar-coverage-and-score-guards/51-02-SUMMARY.md`.
- Found `src/sol_execbench/core/scoring/amd_score.py`.
- Found task commits `70bf2c7` and `8bce00a` in git history.

---
*Phase: 51-sidecar-coverage-and-score-guards*
*Completed: 2026-05-23*
