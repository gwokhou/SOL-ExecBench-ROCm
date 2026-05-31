---
phase: 86-dataset-runner-hardening-integration
verified: 2026-05-31T10:39:43Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 9/9
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 86: Dataset Runner Hardening Integration Verification Report

**Phase Goal:** Dataset runner executions reuse closure helpers to classify resume, reuse, skipped, failed, missing-evidence, and unattempted outcomes deterministically while preserving existing default execution behavior.
**Verified:** 2026-05-31T10:39:43Z
**Status:** passed
**Re-verification:** Yes - after fix commit `90a60aa` and clean targeted re-review `3ce6bc5`.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Resume/reuse requires matching ready-subset, readiness, manifest, problem/workload, solution, and evidence provenance. | VERIFIED | `scripts/run_dataset.py` builds current provenance, reads prior closure provenance from the resolved `execution_closure_path`, and calls `compare_execution_closure_provenance` before authorizing closure-mode existing-pass reuse. Mismatch tests cover stale/missing and divergent provenance. |
| 2 | Existing passing traces become `skipped_existing_pass` only with matching prior provenance and no `--rerun`. | VERIFIED | The existing-pass branch only records `skipped_existing_pass` after empty reuse mismatches; `test_execution_closure_existing_pass_requires_matching_provenance` asserts reuse without invoking `run_cli`. |
| 3 | Default non-closure dataset resume behavior is preserved. | VERIFIED | Fix commit `90a60aa` makes `execution_closure_path` `None` unless `--execution-closure` is supplied or `--ready-subset` enables default closure output. `test_dataset_run_existing_pass_without_closure_keeps_default_skip` covers the ordinary skip path. |
| 4 | Custom `--execution-closure` paths authorize reuse from that exact sidecar. | VERIFIED | The reuse branch reads `_prior_closure_provenance(execution_closure_path)`, not a hard-coded output sidecar. `test_execution_closure_custom_path_authorizes_existing_pass_reuse` writes matching provenance to a custom path and gets `skipped_existing_pass`. |
| 5 | `--rerun` forces a fresh attempted result even when passing traces already exist. | VERIFIED | The reuse branch is bypassed when `args.rerun` is true; `test_execution_closure_rerun_attempts_existing_pass` asserts a fresh attempted closure result. |
| 6 | Missing or mismatched prior provenance is diagnostic, not a reusable clean pass. | VERIFIED | `_prior_closure_provenance` emits `stale_provenance` for absent/unreadable/missing provenance; mismatch reason codes are serialized in `provenance_mismatches` and tests assert fresh attempted output. |
| 7 | Closure output classifies CLI failures, timeouts, nonzero exits, correctness failures, missing traces, missing derived evidence, filtered, and not-attempted workloads. | VERIFIED | Runner maps no-output/nonzero/timeout to `attempted_failed`, missing selected traces to `missing_trace`, evidence gaps to `derived_evidence_missing`, caps/missing rows to `filtered`, and readiness blockers to `not_attempted`; targeted tests cover these states. |
| 8 | Failure diagnostics use stable status/reason vocabulary and bounded relative refs. | VERIFIED | `ExecutionClosureStatus` and `ExecutionClosureReasonCode` are strict enums; `cli_log_ref` is relative. Fix commit `90a60aa` caps CLI log streams with `_CLI_LOG_LIMIT` and `_cli_failure_notes()` reads only the header line. Bounded failure and timeout log tests pass. |
| 9 | Closure writes are deterministic, checksummed, bounded, and sidecar-only. | VERIFIED | `build_execution_closure_report` sorts records and computes checksum; guardrail tests keep closure fields/statuses/reasons out of canonical public payloads; bounded-ref tests assert no tmp absolute paths or raw stdout/stderr in closure JSON. |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `scripts/run_dataset.py` | Provenance-gated reuse plus failure/missing-evidence classification | VERIFIED | Imports closure helpers, resolves closure path correctly, preserves ordinary non-closure skip, checks custom closure provenance, writes bounded log refs and deterministic closure output. |
| `src/sol_execbench/core/dataset/execution_closure.py` | Stable closure schema, statuses, reason codes, provenance comparison, deterministic write helpers | VERIFIED | Strict Pydantic models, status/reason enums, `compare_execution_closure_provenance`, deterministic totals/checksum helpers present and tested. |
| `tests/sol_execbench/test_run_dataset_execution_closure.py` | CPU-safe runner coverage for Phase 86 and clean re-review fixes | VERIFIED | Covers provenance-safe reuse, default non-closure resume, custom closure path reuse, rerun, no-output, nonzero, timeout, bounded logs, missing trace/evidence, filtered, not-attempted, sorted, and bounded-ref behavior. |
| `tests/sol_execbench/test_execution_closure_contract.py` | Contract coverage for helper/vocabulary behavior | VERIFIED | Covers status/reason vocabulary, totals, provenance mismatches, deterministic checksum/write behavior, and strict model validation. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Sidecar-only guardrails | VERIFIED | `test_v1_11_execution_closure_fields_remain_sidecar_only` forbids execution-closure strings in canonical public payloads. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `scripts/run_dataset.py` | `src/sol_execbench/core/dataset/execution_closure.py` | Imports and calls closure helper APIs | VERIFIED | `compare_execution_closure_provenance`, `write_execution_closure_report`, `ExecutionClosureStatus`, and `ExecutionClosureReasonCode` are imported and used by runner closure paths. |
| `scripts/run_dataset.py` | `tests/sol_execbench/test_run_dataset_execution_closure.py` | Monkeypatched CPU-safe runner scenarios | VERIFIED | Tests exercise runner branches without ROCm/Docker/hardware and assert closure JSON outcomes. The generated key-link checker has one expected false negative for plan 86-01 because production code should not import its tests. |
| `scripts/run_dataset.py` | `tests/sol_execbench/test_public_contract_guardrails.py` | Sidecar-only public contract test | VERIFIED | Guardrail test covers Phase 86 closure statuses, reason strings, refs, and checksum fields as sidecar-only diagnostics. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `scripts/run_dataset.py` | prior closure provenance | Resolved `execution_closure_path` parsed by `_prior_closure_provenance` | Yes | VERIFIED - custom and default closure paths flow into provenance comparison before reuse. |
| `scripts/run_dataset.py` | closure records | selected ready workloads, trace payloads, CLI results, evidence sidecars, readiness data | Yes | VERIFIED - records are built per selected workload and written through strict closure report helpers. |
| `scripts/run_dataset.py` | bounded diagnostics | per-problem CLI log file | Yes | VERIFIED - raw stdout/stderr are capped in log files, while closure JSON stores only `cli_log_ref` plus concise notes. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Phase 86 CPU-safe integration, contract, and guardrail suite | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_execution_closure_contract.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_11_execution_closure_fields_remain_sidecar_only -q` | `26 passed in 2.47s` | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| None declared | `find scripts -path '*/tests/probe-*.sh' -type f`; phase plan/summary grep for `probe-*.sh` | No Phase 86 probes found or declared | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| RUNNER-01 | 86-01 | Enforce provenance consistency for resume/reuse | SATISFIED | Reuse path compares current and prior closure provenance; tests cover missing, mismatched, default-path, and custom-path behavior. |
| RUNNER-02 | 86-01 | `skipped_existing_pass` only on matching provenance; `--rerun` reattempts | SATISFIED | Existing pass reuse requires no rerun plus empty mismatch list; tests cover match and rerun behavior. |
| RUNNER-03 | 86-02 | Classify failure/missing/skipped/unattempted states with stable reason codes and bounded log refs | SATISFIED | Runner classification tests cover no-output, nonzero, timeout, missing trace, missing evidence, filtered, and not-attempted paths. |
| RUNNER-04 | 86-01, 86-02 | Preserve existing default execution behavior unless diagnostics are required | SATISFIED | Default non-closure existing pass still skips without requiring `execution_closure.json`; fresh ready-subset tests continue to produce attempted outcomes. |
| RUNNER-05 | 86-01, 86-02 | Deterministic bounded closure writes without sensitive/unbounded payloads | SATISFIED | Checksum/sorting, bounded log, bounded ref, and public contract guardrail tests pass. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| `scripts/run_dataset.py` | 723, 732 | `return {}` | INFO | Normal empty-map returns for absent ready-subset/readiness inputs; not a stub and not user-visible hollow data. |

### Human Verification Required

None. Phase 86 behavior is CPU-safe and covered by automated tests. No new hardware, Docker, ROCm, or external-service verification is required for this phase goal.

### Gaps Summary

No gaps found. The latest code after `90a60aa` satisfies the Phase 86 roadmap success criteria, closes the clean re-review items documented in `3ce6bc5`, preserves default non-closure resume behavior, supports custom closure-path reuse, and keeps diagnostics bounded and sidecar-only.

---

_Verified: 2026-05-31T10:39:43Z_
_Verifier: the agent (gsd-verifier)_
