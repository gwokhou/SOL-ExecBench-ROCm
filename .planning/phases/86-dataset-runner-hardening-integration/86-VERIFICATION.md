---
phase: 86-dataset-runner-hardening-integration
verified: 2026-05-31T10:24:44Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
---

# Phase 86: Dataset Runner Hardening Integration Verification Report

**Phase Goal:** Dataset runner executions reuse closure helpers to classify resume, reuse, skipped, failed, missing-evidence, and unattempted outcomes deterministically while preserving existing default execution behavior.
**Verified:** 2026-05-31T10:24:44Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Resume/reuse requires matching ready-subset, readiness, manifest, problem/workload, solution, and evidence provenance. | VERIFIED | `scripts/run_dataset.py` builds current provenance, reads prior `output/execution_closure.json`, and compares it with `compare_execution_closure_provenance` before existing-pass reuse. Tests cover missing and mismatched prior provenance. |
| 2 | Existing passing traces become `skipped_existing_pass` only with matching prior provenance and no `--rerun`. | VERIFIED | Existing-pass branch is guarded by `if not args.rerun and traces_path.exists()` and emits skipped records only when reuse mismatches are empty. `test_execution_closure_existing_pass_requires_matching_provenance` asserts `run_cli` is not called and closure status is `skipped_existing_pass`. |
| 3 | `--rerun` forces a fresh attempted result even when passing traces already exist. | VERIFIED | Runner prints rerun path and clears previous output before invoking `run_cli`; `test_execution_closure_rerun_attempts_existing_pass` covers the behavior. |
| 4 | Missing or mismatched prior provenance is diagnostic, not a reusable clean pass. | VERIFIED | `_prior_closure_provenance` returns `stale_provenance` for missing/unreadable/missing provenance. Mismatch reason codes are written through `provenance_mismatches`; tests assert fresh attempted output plus mismatch diagnostics. |
| 5 | Closure output classifies CLI failures, timeouts, nonzero exits, correctness failures, missing traces, missing derived evidence, filtered, and not-attempted workloads. | VERIFIED | Runner maps no-output/nonzero/timeout to `attempted_failed`, missing selected traces to `missing_trace`, evidence gaps to `derived_evidence_missing`, caps/missing rows to `filtered`, and readiness blockers to `not_attempted`. Targeted tests cover these paths. |
| 6 | Failure diagnostics use stable status/reason vocabulary and bounded relative refs. | VERIFIED | `ExecutionClosureStatus` and `ExecutionClosureReasonCode` are strict enums; `cli_log_ref` is relative and notes summarize first-line log metadata only. Tests assert raw stdout/stderr and tmp paths are absent from closure JSON. |
| 7 | Existing fresh default execution behavior is preserved except diagnostic sidecar classification. | VERIFIED | Fresh ready-subset tests still call `run_cli`, record `attempted_passed`, and do not require prior closure provenance. Requested Phase 86 suite passes. |
| 8 | Closure writes are deterministic and checksummed. | VERIFIED | `build_execution_closure_report` sorts records/keys, computes checksum, and `test_report_checksum_ignores_checksum_field_and_changes_with_content` plus sorted-record tests cover deterministic output. |
| 9 | Closure fields and reason/status strings remain sidecar-only, not canonical public schema fields. | VERIFIED | `test_v1_11_execution_closure_fields_remain_sidecar_only` forbids closure fields/statuses/reasons in Definition, Workload, and Trace payloads; requested suite passes. |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `scripts/run_dataset.py` | Provenance-gated reuse and failure/missing-evidence classification | VERIFIED | Imports closure helpers, compares prior provenance, records mismatches, uses relative refs, and emits closure statuses for reuse/failure/missing/skipped/unattempted paths. |
| `src/sol_execbench/core/dataset/execution_closure.py` | Stable closure schema, statuses, reason codes, provenance comparison, deterministic write helpers | VERIFIED | Strict Pydantic models, status/reason enums, `compare_execution_closure_provenance`, deterministic JSON/checksum helpers present and tested. |
| `tests/sol_execbench/test_run_dataset_execution_closure.py` | CPU-safe runner coverage for Phase 86 behavior | VERIFIED | Contains provenance-safe reuse, rerun, no-output, nonzero, timeout, missing trace, missing evidence, filtered, not-attempted, sorted, and bounded-ref tests. |
| `tests/sol_execbench/test_execution_closure_contract.py` | Contract coverage for helper/vocabulary behavior | VERIFIED | Covers status/reason vocabulary, totals, provenance mismatches, deterministic checksum/write behavior, and model strictness. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Sidecar-only guardrails | VERIFIED | `test_v1_11_execution_closure_fields_remain_sidecar_only` forbids execution-closure strings in canonical public payloads. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `scripts/run_dataset.py` | `src/sol_execbench/core/dataset/execution_closure.py` | Imports and calls closure helper APIs | VERIFIED | `build_execution_closure_report`, `write_execution_closure_report`, `compare_execution_closure_provenance`, status/reason enums, and evidence helpers are used by runner closure paths. |
| `scripts/run_dataset.py` | `tests/sol_execbench/test_run_dataset_execution_closure.py` | Monkeypatched `run_cli`/`subprocess.run` scenarios | VERIFIED | Tests exercise runner branches without ROCm/Docker/hardware and assert closure JSON outcomes. The generated key-link checker reported one false negative because production code should not import its tests. |
| `scripts/run_dataset.py` | `tests/sol_execbench/test_public_contract_guardrails.py` | Sidecar-only public contract test | VERIFIED | Guardrail test covers Phase 86 status/reason strings as sidecar-only diagnostics. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `scripts/run_dataset.py` | prior closure provenance | `output_dir / "execution_closure.json"` parsed by `_prior_closure_provenance` | Yes | VERIFIED - parsed provenance is compared before reuse; stale/unreadable input produces `stale_provenance`. |
| `scripts/run_dataset.py` | closure records | selected ready workloads, trace payloads, CLI results, evidence sidecars, readiness data | Yes | VERIFIED - records are built per selected workload and written through strict closure report helpers. |
| `scripts/run_dataset.py` | bounded diagnostics | per-problem CLI log file | Yes | VERIFIED - closure stores `cli_log_ref` plus concise notes, not raw log bodies. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Phase 86 CPU-safe integration suite | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_execution_closure_contract.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_11_execution_closure_fields_remain_sidecar_only -q` | `22 passed in 2.41s` | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| None declared | `find scripts -path '*/tests/probe-*.sh' -type f` and phase artifact grep | No probes found or declared for Phase 86 | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| RUNNER-01 | 86-01 | Enforce provenance consistency for resume/reuse | SATISFIED | Reuse path compares current and prior closure provenance; tests cover missing/mismatched provenance. |
| RUNNER-02 | 86-01 | `skipped_existing_pass` only on matching provenance; `--rerun` reattempts | SATISFIED | Existing pass branch requires no rerun plus empty mismatch list; tests cover match and rerun behavior. |
| RUNNER-03 | 86-02 | Classify failure/missing/skipped/unattempted states with stable reason codes and bounded log refs | SATISFIED | Runner classification tests cover no-output, nonzero, timeout, missing trace, missing evidence, filtered, and not-attempted paths. |
| RUNNER-04 | 86-01, 86-02 | Preserve existing default execution behavior unless diagnostics are required | SATISFIED | Fresh ready-subset tests continue to use attempted execution and pass in the requested suite. |
| RUNNER-05 | 86-01, 86-02 | Deterministic bounded closure writes without sensitive/unbounded payloads | SATISFIED | Deterministic checksum/sorting tests and bounded-ref tests pass; closure JSON excludes tmp paths and raw logs. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| `scripts/run_dataset.py` | 714, 723 | `return {}` | INFO | Normal empty-map returns for absent ready subset/readiness inputs; not a stub and not user-visible hollow data. |

### Human Verification Required

None. Phase 86 behavior is CPU-safe and covered by automated tests; no visual, external-service, or hardware UAT is required for the phase goal.

### Gaps Summary

No gaps found. All roadmap success criteria, plan must-haves, key implementation links, requirements RUNNER-01 through RUNNER-05, and the requested verification command passed against the codebase.

---

_Verified: 2026-05-31T10:24:44Z_
_Verifier: the agent (gsd-verifier)_
