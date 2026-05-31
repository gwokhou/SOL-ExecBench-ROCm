---
phase: 86-dataset-runner-hardening-integration
plan: 02
subsystem: dataset-runner
tags: [execution-closure, failure-classification, bounded-logs, pytest]
requires:
  - phase: 86-dataset-runner-hardening-integration
    provides: provenance-gated existing-pass reuse
provides:
  - Deterministic CLI nonzero and timeout classification as attempted_failed
  - Bounded relative cli_log_ref diagnostics for runner failures
  - CPU-safe tests for missing trace, missing evidence, filtered, not_attempted, and public sidecar guardrails
affects: [dataset-runner, execution-closure, public-contract-guardrails]
tech-stack:
  added: []
  patterns: [bounded diagnostic log refs, concise closure notes, CPU-safe subprocess monkeypatch tests]
key-files:
  created: [.planning/phases/86-dataset-runner-hardening-integration/86-02-SUMMARY.md]
  modified:
    - scripts/run_dataset.py
    - tests/sol_execbench/test_run_dataset_execution_closure.py
    - tests/sol_execbench/test_execution_closure_contract.py
    - tests/sol_execbench/test_public_contract_guardrails.py
key-decisions:
  - "Raw CLI stdout/stderr remains in per-problem log files; execution_closure.json stores only relative cli_log_ref plus concise notes."
  - "run_cli preserves its public return shape while treating nonzero exits and timeouts as None so closure records classify them as attempted_failed."
patterns-established:
  - "Runner failure tests monkeypatch subprocess.run or run_cli and assert closure JSON excludes raw logs and absolute temp paths."
requirements-completed: [RUNNER-03, RUNNER-04, RUNNER-05]
duration: 3min
completed: 2026-05-31
---

# Phase 86 Plan 02: Runner Failure Classification Summary

**Runner closure records now classify nonzero CLI exits, timeouts, no-output failures, missing traces, and missing evidence with bounded sidecar diagnostics**

## Performance

- **Duration:** 3 min
- **Started:** 2026-05-31T10:16:00Z
- **Completed:** 2026-05-31T10:19:05Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added RED coverage for nonzero CLI exits and subprocess timeouts that previously were not deterministically classified.
- Updated `run_cli` to catch timeouts, reject nonzero exits even when stdout contains trace JSON, and write bounded per-problem log files.
- Kept closure JSON bounded by storing relative `cli_log_ref` values and concise notes only.
- Verified missing traces, missing derived evidence, filtered workloads, readiness-blocked workloads, and public sidecar-only guardrails.

## Task Commits

1. **Task 1: Specify bounded failure and missing-evidence classification** - `5f38abb` (test)
2. **Task 2: Implement deterministic classification in runner closure records** - `f19095b` (feat)
3. **Task 3: Run CPU-safe integration guardrails** - verified by final CPU-safe suite

## Files Created/Modified

- `scripts/run_dataset.py` - Classifies CLI timeouts and nonzero exits as no-trace runner failures with bounded log files and concise closure notes.
- `tests/sol_execbench/test_run_dataset_execution_closure.py` - Adds CPU-safe subprocess monkeypatch tests for nonzero and timeout failures plus bounded output assertions.
- `tests/sol_execbench/test_execution_closure_contract.py` - Reused existing closure vocabulary and totals contract coverage.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Reused sidecar-only guardrail coverage for execution closure strings.

## Decisions Made

- No new closure status or reason vocabulary was required; nonzero and timeout runner failures fit the existing `attempted_failed` sidecar status.
- Failure details are summarized from the first line of the bounded log file so raw stdout/stderr never flows into `ExecutionClosureRecord.notes`.

## Deviations from Plan

None - plan executed as written.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py::test_execution_closure_classifies_cli_nonzero_exit_with_bounded_log_ref tests/sol_execbench/test_run_dataset_execution_closure.py::test_execution_closure_classifies_cli_timeout_with_bounded_log_ref -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_execution_closure_contract.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_execution_closure_contract.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_11_execution_closure_fields_remain_sidecar_only -q`

## Known Stubs

None.

## Threat Flags

None.

## Issues Encountered

The initial Plan 86-02 no-output and missing-trace tests already passed because the current runner had partial coverage. Additional RED tests exposed the remaining nonzero-exit and timeout gaps before implementation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 86 is complete. The runner now provides provenance-safe reuse and bounded closure diagnostics without adding ROCm, Docker, network, dependency, or primary CLI requirements.

## Self-Check: PASSED

- Found summary file: `.planning/phases/86-dataset-runner-hardening-integration/86-02-SUMMARY.md`
- Found task commits: `5f38abb`, `f19095b`
- Verification suite passed: 22 CPU-safe tests

---
*Phase: 86-dataset-runner-hardening-integration*
*Completed: 2026-05-31*
