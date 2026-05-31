---
phase: 86-dataset-runner-hardening-integration
reviewed: 2026-05-31T10:29:27Z
depth: deep
files_reviewed: 3
files_reviewed_list:
  - scripts/run_dataset.py
  - tests/sol_execbench/test_run_dataset_execution_closure.py
  - tests/sol_execbench/test_public_contract_guardrails.py
findings:
  critical: 2
  warning: 1
  info: 0
  total: 3
status: findings_found
---

# Phase 86: Code Review Report

**Reviewed:** 2026-05-31T10:29:27Z
**Depth:** deep
**Files Reviewed:** 3
**Status:** findings_found

## Summary

Reviewed Phase 86 runner hardening changes for provenance-gated reuse, rerun behavior, failure classification, bounded refs, sidecar-only contracts, and tests. The CPU-safe verification suite still passes, but the implementation has a default runner behavior regression and a custom closure path reuse bug. Diagnostic log handling also remains unbounded despite the phase's bounded-log requirement.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Provenance gate forces ordinary non-closure runs to rerun passing traces

**File:** `scripts/run_dataset.py:1517`

**Severity:** BLOCKER

**Impact:** The Phase 86 provenance check now runs for every existing passing `traces.json`, even when the user is doing a normal dataset run without `--ready-subset` or `--execution-closure`. In that default mode no `output/execution_closure.json` is produced, so `_prior_closure_provenance()` always returns `stale_provenance` and the runner reruns an already-passing problem instead of preserving the historical "Skipping (already passed)" behavior. This violates RUNNER-04 and can turn a cheap resume into a full GPU reevaluation.

**Suggested Fix:** Only require prior closure provenance when closure/reuse diagnostics are actually in scope, e.g. when `execution_closure_path is not None` or the run is ready-subset bounded. Preserve the old skip branch for normal non-closure runs.

```python
if summary["failed"] == 0:
    if execution_closure_path is not None:
        prior_provenance, stale_mismatch = _prior_closure_provenance(
            _prior_execution_closure_path(execution_closure_path, output_dir)
        )
        reuse_mismatches = ...
        if reuse_mismatches:
            provenance_mismatches.extend(reuse_mismatches)
            print("  Re-running (previous pass has stale closure provenance).")
        else:
            skip_existing_pass()
            continue
    else:
        print("  Skipping (already passed). Use --rerun to re-evaluate.")
        summaries.append(summary)
        continue
```

Add a regression test that runs without `--ready-subset` and without `--execution-closure`, pre-creates an all-passing `traces.json`, monkeypatches `run_cli` to fail if called, and asserts the old skip behavior remains intact.

### CR-02: Custom `--execution-closure` paths are ignored when authorizing reuse

**File:** `scripts/run_dataset.py:1521`

**Severity:** BLOCKER

**Impact:** The reuse check always reads `output_dir / "execution_closure.json"` instead of the actual `execution_closure_path` computed from `--execution-closure`. A user who writes the closure sidecar to a custom path cannot get provenance-safe reuse on the next run with the same custom path; the runner treats the valid prior sidecar as missing/stale and reruns the workload. Conversely, the code may consult a stale default sidecar even when the user explicitly selected a different closure report location for the run.

**Suggested Fix:** Use the resolved `execution_closure_path` for provenance reuse whenever it exists, with an explicit fallback to the default output sidecar only when no custom path is configured.

```python
def _prior_execution_closure_path(
    execution_closure_path: Path | None,
    output_dir: Path,
) -> Path:
    return execution_closure_path or (output_dir / "execution_closure.json")

prior_provenance, stale_mismatch = _prior_closure_provenance(
    _prior_execution_closure_path(execution_closure_path, output_dir)
)
```

Add a test that writes a matching prior closure to a custom path, runs with `--execution-closure <custom>`, and asserts `run_cli` is not called and `skipped_existing_pass` is recorded.

## Warnings

### WR-01: CLI failure log files and note extraction are unbounded

**File:** `scripts/run_dataset.py:333`

**Severity:** WARNING

**Impact:** `_save_cli_log()` and `_save_cli_timeout_log()` write full stdout/stderr bodies to disk, and `_cli_failure_notes()` reads the entire log file just to inspect the first line. Phase 86 explicitly called for bounded logs/refs and a DoS mitigation for unbounded diagnostics. A failed subprocess can still emit arbitrarily large stdout/stderr, consuming disk and memory during failure classification. The closure JSON references are bounded, but the log artifacts themselves are not.

**Suggested Fix:** Cap captured log content at a fixed byte/character limit when writing failure logs, append an explicit truncation marker, and read only the first line or first small chunk when generating closure notes.

```python
_CLI_LOG_LIMIT = 64 * 1024

def _bounded_stream(value: str | bytes | None) -> str:
    text = _timeout_stream_text(value)
    if len(text) <= _CLI_LOG_LIMIT:
        return text
    return text[:_CLI_LOG_LIMIT] + "\n[truncated]"
```

Use `_bounded_stream()` for stdout/stderr in both log writers and avoid `cli_log.read_text()` in `_cli_failure_notes()`.

---

_Reviewed: 2026-05-31T10:29:27Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
