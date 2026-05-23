---
phase: 52-dataset-runner-and-public-contract-closure
plan: 03
subsystem: testing
tags: [pytest, public-contract, claim-guardrails, solar-derivation, amd-native-score]

requires:
  - phase: 52-dataset-runner-and-public-contract-closure
    provides: "Plans 52-01 and 52-02 added derived SOLAR report refs and documented sidecar claim boundaries."
provides:
  - "TEST-04 exact-key guardrails for canonical schemas, primary CLI help, canonical trace JSONL, public evidence_refs, and derived-report-only refs."
  - "TEST-05 claim guardrails for v1.10 no-claim language and positive overclaim rejection."
  - "Full Phase 52 focused regression and Ruff gate results."
affects: [release-closure, public-contracts, docs-claims, dataset-runner]

tech-stack:
  added: []
  patterns:
    - "Use exact recursive JSON key assertions for public contract boundaries."
    - "Use phrase-context claim checks instead of broad token deny lists."

key-files:
  created:
    - ".planning/phases/52-dataset-runner-and-public-contract-closure/52-03-SUMMARY.md"
  modified:
    - "tests/sol_execbench/test_public_contract_guardrails.py"
    - "tests/sol_execbench/test_v1_9_validation_closure.py"

key-decisions:
  - "Keep Phase 52 derived refs in derived_evidence_refs while preserving established public evidence_refs keys."
  - "Normalize whitespace in documentation claim guardrails so Markdown wrapping does not weaken or over-trigger claim checks."

patterns-established:
  - "Public contract tests should assert exact key sets for canonical model surfaces."
  - "Claim tests should include synthetic positive overclaim rejection plus real docs no-claim requirements."

requirements-completed: [REPORT-04, TEST-04, TEST-05]

duration: 31min
completed: 2026-05-23
---

# Phase 52 Plan 03: Public Contract And Claim Guardrail Closure Summary

**Exact public contract and v1.10 claim-boundary guardrails with full Phase 52 regression closure.**

## Performance

- **Duration:** 31 min
- **Started:** 2026-05-23T10:54:58Z
- **Completed:** 2026-05-23T11:25:36Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added TEST-04 exact-key guardrails for canonical `Definition`, `Workload`, `Trace`, canonical trace JSONL, primary CLI help, public score `evidence_refs`, and derived-report-only `derived_evidence_refs`.
- Added TEST-05 v1.10 claim guardrails that reject positive overclaims for paper parity, paper-scale extraction, B200/Blackwell equivalence, hosted leaderboard readiness, and deferred hardware validation claims.
- Ran the full Phase 52 pytest and Ruff closure gates successfully.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend exact-key public contract guardrails** - `c00c4b6` (`#52 - Add public contract guardrails for derived reports`)
2. **Task 2: Extend claim guardrails for v1.10 artifacts and docs** - `dceed88` (`#52 - Add v1.10 claim boundary guardrails`)
3. **Task 3: Run full Phase 52 regression closure** - `ff69fd4` (`#52 - Close Phase 52 regression gate`)

## Files Created/Modified

- `tests/sol_execbench/test_public_contract_guardrails.py` - Adds exact canonical key and derived evidence boundary checks.
- `tests/sol_execbench/test_v1_9_validation_closure.py` - Adds v1.10 positive-overclaim rejection and no-claim context checks.
- `.planning/phases/52-dataset-runner-and-public-contract-closure/52-03-SUMMARY.md` - Records plan closure.

## Decisions Made

- Kept guardrails as tests only; no runner, schema, CLI, dependency, hardware, or candidate-execution behavior changed.
- Used exact key assertions for JSON/public contract surfaces while preserving valid sidecar-only fields such as `coverage_summary`.
- Used whitespace-normalized phrase-context checks for documentation so historical/deferred mentions remain allowed when framed as no-claims.

## Deviations from Plan

None - plan scope was executed without adding runner or docs features beyond guardrails.

## Issues Encountered

- Task 1 TDD note: added guardrails passed immediately because the existing implementation already satisfied those exact-key assertions.
- Task 2 RED gate: the initial claim-context assertion failed because required wording was split across Markdown line wrapping. The guardrail was corrected to normalize whitespace, then passed.

## Verification

- `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_run_dataset_amd_score.py -k "schema or cli or trace or evidence or public or solar or derived" -n 0 -x` - 28 passed.
- `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_v1_9_validation_closure.py -k "claim or boundary or paper or leaderboard or validation or B200 or Blackwell or MI300X or NVFP4 or MXFP4" -n 0 -x` - 25 passed.
- `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_v1_9_validation_closure.py tests/sol_execbench/test_solar_derivation_evidence.py -n 0` - 168 passed.
- `uv run --with ruff ruff check scripts/run_dataset.py src/sol_execbench/core/scoring/amd_score.py src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_v1_9_validation_closure.py tests/sol_execbench/test_solar_derivation_evidence.py` - passed.

## Known Stubs

None. The `None` and empty dict/list values found during stub scan are test fixture inputs, not UI or runtime placeholders.

## Threat Flags

None. This plan added tests only and introduced no new endpoints, auth paths, file access trust boundaries, schema fields, dependencies, or candidate-execution paths.

## TDD Gate Compliance

- Task 1 did not produce a failing RED result because the new guardrails passed against the current implementation.
- Task 2 produced and resolved a RED result before passing.
- Task 3 was a verification-only task and was recorded with an empty signed commit.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 52 public contract and claim guardrails are closed. Remaining deferred work stays outside v1.10 scope: paper-scale extraction, hosted leaderboard readiness, Blackwell/B200 equivalence, CDNA3/MI300X/CDNA4 hardware validation, and NVFP4/MXFP4 validation.

## Self-Check: PASSED

- Found `tests/sol_execbench/test_public_contract_guardrails.py`.
- Found `tests/sol_execbench/test_v1_9_validation_closure.py`.
- Found `.planning/phases/52-dataset-runner-and-public-contract-closure/52-03-SUMMARY.md`.
- Found commits `c00c4b6`, `dceed88`, and `ff69fd4` in git history.

---
*Phase: 52-dataset-runner-and-public-contract-closure*
*Completed: 2026-05-23*
