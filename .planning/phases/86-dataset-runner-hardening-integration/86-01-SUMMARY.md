---
phase: 86-dataset-runner-hardening-integration
plan: 01
subsystem: dataset-runner
tags: [execution-closure, provenance, reuse, pytest]
requires:
  - phase: 83-closure-contracts-and-provenance-foundation
    provides: execution closure sidecar schema and provenance comparison helpers
provides:
  - Provenance-gated skipped_existing_pass reuse for ready-subset dataset runs
  - Fresh attempted execution when prior closure provenance is missing or mismatched
  - CPU-safe regression tests for --rerun and stale existing-pass behavior
affects: [dataset-runner, execution-closure, public-contract-guardrails]
tech-stack:
  added: []
  patterns: [sidecar-only provenance diagnostics, CPU-safe monkeypatched runner tests]
key-files:
  created: [.planning/phases/86-dataset-runner-hardening-integration/86-01-SUMMARY.md]
  modified:
    - scripts/run_dataset.py
    - tests/sol_execbench/test_run_dataset_execution_closure.py
    - tests/sol_execbench/test_public_contract_guardrails.py
key-decisions:
  - "Existing passing traces authorize skipped_existing_pass only when output/execution_closure.json contains matching provenance."
  - "Missing or unreadable prior closure provenance is recorded as stale_provenance and causes a fresh attempted run."
patterns-established:
  - "Runner reuse checks compare the current expected closure provenance against the prior sidecar before using old traces."
requirements-completed: [RUNNER-01, RUNNER-02, RUNNER-04, RUNNER-05]
duration: 6min
completed: 2026-05-31
---

# Phase 86 Plan 01: Provenance-Gated Existing-Pass Reuse Summary

**Existing passing traces are reused only when prior execution-closure provenance matches the current selected run configuration**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-31T10:09:50Z
- **Completed:** 2026-05-31T10:15:48Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added CPU-safe tests proving old passing traces without prior closure provenance are freshly attempted.
- Added runner-side provenance comparison before emitting `skipped_existing_pass`.
- Preserved default fresh ready-subset behavior and `--rerun` fresh attempt behavior.
- Extended sidecar-only public contract guardrails for closure provenance reason strings.

## Task Commits

1. **Task 1: Specify provenance-gated existing-pass reuse** - `521e445` (test)
2. **Task 2: Enforce safe existing-output reuse** - `c82dbbd` (feat)
3. **Task 3: Guard default behavior and public contract boundaries** - `0a31d53` (test)

## Files Created/Modified

- `scripts/run_dataset.py` - Loads prior closure provenance, compares reuse inputs, records stale/mismatch diagnostics, and bypasses reuse for `--rerun`.
- `tests/sol_execbench/test_run_dataset_execution_closure.py` - Adds CPU-safe regression coverage for matching/missing/mismatched prior provenance and rerun attempts.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Keeps execution-closure reason/status strings out of canonical Definition/Workload/Trace payloads.

## Decisions Made

- Prior closure reuse is read from `output/execution_closure.json`, matching the existing output tree whose traces are being considered for reuse.
- Missing, unreadable, or provenance-less prior closure reports use the existing `stale_provenance` reason code instead of adding new vocabulary.

## Deviations from Plan

None - plan executed as written.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py::test_execution_closure_existing_pass_requires_matching_provenance tests/sol_execbench/test_run_dataset_execution_closure.py::test_execution_closure_existing_pass_without_prior_provenance_runs_fresh tests/sol_execbench/test_run_dataset_execution_closure.py::test_execution_closure_existing_pass_mismatched_provenance_runs_fresh tests/sol_execbench/test_run_dataset_execution_closure.py::test_execution_closure_rerun_attempts_existing_pass tests/sol_execbench/test_execution_closure_contract.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_execution_closure_contract.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_execution_closure_contract.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_11_execution_closure_fields_remain_sidecar_only -q`

## Known Stubs

None.

## Threat Flags

None.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 86-02 can build on the same closure record path to classify runner failures, missing traces, and missing derived evidence with bounded diagnostics.

## Self-Check: PASSED

- Found summary file: `.planning/phases/86-dataset-runner-hardening-integration/86-01-SUMMARY.md`
- Found task commits: `521e445`, `c82dbbd`, `0a31d53`

---
*Phase: 86-dataset-runner-hardening-integration*
*Completed: 2026-05-31*
