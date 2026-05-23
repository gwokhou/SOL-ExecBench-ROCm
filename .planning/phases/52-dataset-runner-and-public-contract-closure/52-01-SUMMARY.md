---
phase: 52-dataset-runner-and-public-contract-closure
plan: 01
subsystem: scoring
tags: [dataset-runner, amd-native-score, solar-derivation, derived-reports]

requires:
  - phase: 51-sidecar-coverage-and-score-guards
    provides: "SOLAR aggregate score guards and sidecar coverage/status payloads"
provides:
  - "Opt-in dataset-runner generated SOLAR derivation sidecars"
  - "AMD-native score report pass-through for explicit generated SOLAR evidence"
  - "Derived report-only evidence refs for formula, hardware model, coverage, and score eligibility"
  - "Skipped existing passing traces still populate requested derived reports and sidecars"
affects: [phase-52, REPORT-04, TEST-04, TEST-05, amd-native-score, dataset-runner]

tech-stack:
  added: []
  patterns:
    - "Generated sidecar pass-through from canonical Definition and Workload inputs"
    - "Derived-report-only metadata separated from public score evidence_refs"

key-files:
  created:
    - ".planning/phases/52-dataset-runner-and-public-contract-closure/52-01-SUMMARY.md"
  modified:
    - "scripts/run_dataset.py"
    - "src/sol_execbench/core/scoring/amd_score.py"
    - "tests/sol_execbench/test_run_dataset_amd_score.py"
    - "tests/sol_execbench/test_amd_native_score.py"

key-decisions:
  - "Generated SOLAR derivation sidecars are built and immediately parsed through the existing sidecar parser before scoring."
  - "Derived report audit refs live in AmdNativeScore.derived_evidence_refs, leaving public evidence_refs keys unchanged."
  - "The dataset-runner skip branch now builds requested derived artifacts before continuing when existing traces already pass."

patterns-established:
  - "Dataset-runner derived artifacts are assembled through a shared helper for fresh and skipped traces."
  - "Absent SOLAR sidecar evidence remains neutral; explicit generated evidence can suppress scoring only through parsed aggregate status."

requirements-completed: [REPORT-04]

duration: 6min
completed: 2026-05-23
---

# Phase 52 Plan 01: Dataset Runner SOLAR Report Integration Summary

**Dataset-runner generated SOLAR sidecars now feed AMD-native derived reports without widening canonical traces or public evidence refs**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-23T10:06:59Z
- **Completed:** 2026-05-23T10:12:35Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Added `--solar-derivation` to `scripts/run_dataset.py` for opt-in generated SOLAR derivation sidecars.
- Built sidecars only from `Definition` and `Workload`, then validated them through `solar_derivation_from_dict()` before passing them to AMD-native scoring.
- Added `derived_evidence_refs` to AMD-native score payloads for formula, hardware model, coverage, and score eligibility audit links while preserving existing `evidence_refs` keys.
- Fixed the existing-trace skip path so already-passing `traces.json` files still produce requested AMD score rows and SOLAR sidecars; failed existing traces still rerun.
- Advanced TEST-04 and TEST-05 coverage with focused regression tests while leaving final public-contract and claim guardrails for Plan 52-03.

## Task Commits

1. **Task 1: Add runner tests for SOLAR sidecars and skipped traces** - `55fee4c` (`#52 - Add failing dataset runner SOLAR report tests`)
2. **Task 2: Wire opt-in SOLAR sidecars into dataset-runner reports** - `65f73da` (`#52 - Wire dataset runner SOLAR derived reports`)
3. **Task 3: Lock skipped-trace derived report behavior** - `c98dbaf` (`#52 - Preserve skipped trace derived report generation`)

## Files Created/Modified

- `scripts/run_dataset.py` - Added `--solar-derivation`, generated/validated SOLAR sidecars, shared derived artifact assembly, and skip-path report generation.
- `src/sol_execbench/core/scoring/amd_score.py` - Added derived-report-only `derived_evidence_refs` propagation and serialization.
- `tests/sol_execbench/test_run_dataset_amd_score.py` - Added sidecar payload, skipped-trace, failed-existing-trace rerun, and helper-level report tests.
- `tests/sol_execbench/test_amd_native_score.py` - Added derived evidence refs contract coverage.
- `.planning/phases/52-dataset-runner-and-public-contract-closure/52-01-SUMMARY.md` - Execution summary.

## Decisions Made

- Used `build_solar_derivation_evidence(Definition, Workload)` plus `solar_derivation_from_dict()` for all generated sidecars; no ad hoc JSON construction and no candidate execution.
- Kept `claim_level` unchanged at `amd-native-derived`.
- Kept canonical `traces.json` unchanged; SOLAR payloads are sidecars and report metadata only.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Verification

- `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py -k "solar or derivation or evidence or sidecar or skipped or skip or claim" -n 0 -x` - 14 passed, 14 deselected.
- `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py -k "skipped or skip or existing or solar or sidecar" -n 0 -x` - 4 passed, 4 deselected.
- `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_solar_derivation_evidence.py -n 0` - 109 passed.
- `uv run --with ruff ruff check scripts/run_dataset.py src/sol_execbench/core/scoring/amd_score.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py` - passed.

## Known Stubs

None. Stub scan only found intentional test `None`/empty fixture values and argparse defaults.

## Threat Flags

None. The planned local filesystem-to-runner sidecar surface was mitigated by generating from canonical inputs, validating through `solar_derivation_from_dict()`, and keeping SOLAR fields out of canonical trace JSON.

## Risks

- Sidecars are generated only by the dataset runner; reading pre-existing SOLAR sidecar directories remains intentionally out of scope.
- Hardware validation and candidate execution remain out of scope for this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 52-02 can document the runner-derived report workflow. Plan 52-03 can add final public contract guardrails against canonical trace, primary CLI, and evidence-ref drift.

## Self-Check: PASSED

- Found `.planning/phases/52-dataset-runner-and-public-contract-closure/52-01-SUMMARY.md`.
- Found task commits `55fee4c`, `65f73da`, and `c98dbaf` in git history.

---
*Phase: 52-dataset-runner-and-public-contract-closure*
*Completed: 2026-05-23*
