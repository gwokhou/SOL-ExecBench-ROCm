---
phase: 47-derivation-contract-and-golden-fixture-matrix
plan: 01
subsystem: docs
tags: [solar, derivation-contract, fixtures, sidecars, guardrails]

requires:
  - phase: 46
    provides: v1.9 AMD SOL/SOLAR sidecar and claim-boundary foundation
provides:
  - Internal sidecar-only SOLAR derivation contract for v1.10
  - Fixture schema vocabulary for later golden matrix plans
  - Claim-boundary inventory for deferred paper-scale, leaderboard, B200, and hardware-validation work
affects: [phase-47, phase-48, phase-49, phase-50, phase-51, phase-52]

tech-stack:
  added: []
  patterns:
    - Internal docs contract defines machine-readable fixture expectations before implementation
    - Sidecar-only derived evidence remains separate from canonical schemas and primary CLI behavior

key-files:
  created:
    - docs/internal/solar_derivation_contract.md
  modified: []

key-decisions:
  - "Documented SOLAR derivation evidence as sidecar-only or opt-in report content, not canonical schema or primary CLI behavior."
  - "Defined degraded and unsupported fixture behavior through expected status, missing evidence, stable warning prefixes, and rationale instead of exceptions."

patterns-established:
  - "Fixture contract uses exact family identifiers: attention, moe, convolution, ssm_mamba, embedding_positional, and linear_projection."
  - "Claim boundaries are explicit non-goals for paper-scale extraction, hosted leaderboard readiness, NVIDIA Blackwell/B200 equivalence, and new real-hardware validation."

requirements-completed: [TEST-01, TEST-02]

duration: 2min
completed: 2026-05-23
---

# Phase 47 Plan 01: Sidecar-Only SOLAR Derivation Contract Summary

**Internal v1.10 SOLAR derivation contract with sidecar-only fixture schema, family/state vocabulary, degradation semantics, and no-claim boundaries**

## Performance

- **Duration:** 2 min
- **Started:** 2026-05-23T04:18:28Z
- **Completed:** 2026-05-23T04:20:03Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created the internal SOLAR derivation contract document for Phase 47.
- Defined the six required TEST-01 family identifiers and the confidence/status vocabulary later fixtures must use.
- Documented positive, degraded, unsupported, and negative fixture expectations through machine-readable fields and stable warning prefixes.
- Preserved sidecar-only boundaries and explicit deferred claim areas.

## Task Commits

Each task was committed atomically:

1. **Task 1: Write the sidecar-only derivation contract** - `a141ccf` (docs)

## Files Created/Modified

- `docs/internal/solar_derivation_contract.md` - Internal v1.10 derivation contract covering sidecar-only artifact rules, fixture schema, family/state vocabulary, golden matrix inventory, degradation behavior, and downstream phase consumption.

## Decisions Made

- Kept this plan documentation-only, with no production extractor, model, score, schema, or CLI changes.
- Used the exact lowercase vocabulary required by the plan: `supported`, `inexact`, `unsupported`, `scored`, `degraded`, and `unscored`.
- Represented negative fixture behavior as valid expectation data, not exception behavior.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Verification

Passed:

```bash
test -f docs/internal/solar_derivation_contract.md
```

Passed:

```bash
rg -n "attention|moe|convolution|ssm_mamba|embedding_positional|linear_projection|expected_family|expected_status|not paper-scale dataset extraction|not hosted leaderboard readiness|not NVIDIA Blackwell/B200 equivalence|not new real-hardware validation" docs/internal/solar_derivation_contract.md
```

No pytest run was required by this plan; executable tests are owned by later Phase 47 plans.

## Known Stubs

None.

## Threat Flags

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 47-02 can add the fixture loader and loader-only schema tests against the schema and vocabulary defined here. Later fixture batches can consume the family, state, warning-prefix, and claim-boundary rules without changing production behavior.

## Self-Check: PASSED

- Found created contract file: `docs/internal/solar_derivation_contract.md`
- Found created summary file: `.planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-01-SUMMARY.md`
- Found task commit: `a141ccf`

---
*Phase: 47-derivation-contract-and-golden-fixture-matrix*
*Completed: 2026-05-23*
