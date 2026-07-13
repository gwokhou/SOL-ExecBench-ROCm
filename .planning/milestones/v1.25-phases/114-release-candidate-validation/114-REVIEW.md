---
phase: 114-release-candidate-validation
reviewed: 2026-06-01T10:14:24Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - scripts/release_candidate_validation.py
  - tests/sol_execbench/test_release_candidate_validation.py
  - docs/internal/release_candidate_validation.md
findings:
  critical: 1
  warning: 2
  info: 0
  total: 3
status: findings_found
---

# Phase 114: Code Review Report

**Reviewed:** 2026-06-01T10:14:24Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** findings_found

## Summary

Reviewed the release-candidate validation wrapper, its focused tests, and maintainer documentation. The implementation correctly avoids shell interpolation for subprocess execution, but the persisted stdout/stderr redaction is too narrow for a shareable release artifact. Dataset-slice evidence also has two correctness gaps around bounded command enforcement and dependent sidecar execution.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Sensitive log redaction misses common credential formats

**File:** `scripts/release_candidate_validation.py:38`
**Classification:** BLOCKER
**Issue:** The wrapper persists subprocess stdout/stderr tails into `release_candidate_validation.json`, but `TOKEN_PATTERN` only redacts bare names like `TOKEN=value` where the sensitive word is immediately followed by `=` or `:`. Common environment and log formats such as `AWS_SECRET_ACCESS_KEY=...`, `HF_TOKEN = ...`, `GITHUB_TOKEN: ...`, and `Authorization: Bearer ...` are not reliably redacted. The exposed `--log-tail-chars` option is also parsed at line 155 but never passed into `_tail()` at lines 254-255, so maintainers cannot reduce or disable captured tails for sensitive runs.
**Fix:**
```python
TOKEN_PATTERN = re.compile(
    r"(?ix)"
    r"("
    r"(?:[A-Z0-9_]*?(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|CREDENTIAL)[A-Z0-9_-]*?)"
    r"|authorization"
    r")"
    r"(\s*[:=]\s*|:\s*bearer\s+)"
    r"([^\s'\"]+)"
)

def _run_check(..., log_tail_chars: int = DEFAULT_LOG_TAIL_CHARS) -> ValidationResult:
    ...
    stdout_tail = _tail(completed.stdout, log_tail_chars)
    stderr_tail = _tail(completed.stderr, log_tail_chars)
```
Add regression tests for spaced assignments, prefixed secret names, authorization headers, and `--log-tail-chars 0`.

## Warnings

### WR-01: Dataset command override can bypass bounded-slice guarantees

**File:** `scripts/release_candidate_validation.py:170`
**Classification:** WARNING
**Issue:** `--dataset-limit` is required, but when `--dataset-command` is supplied the script accepts the override verbatim and still records `dataset_limit` and the "not full 235-problem paper validation" boundary at lines 206-208. A release run can therefore execute an unbounded or stale dataset command while the summary claims bounded validation. This undermines the phase requirement to reject or document unbounded paper-scale attempts.
**Fix:** Either remove the public dataset command override, or validate any override before execution:
```python
if args.dataset_command:
    command = list(args.dataset_command)
    if "--limit" not in command or str(args.dataset_limit) not in command:
        raise SystemExit("--dataset-command must include the requested bounded --limit")
    if "--rerun" not in command:
        raise SystemExit("--dataset-command must include --rerun")
    if "--execution-closure" not in command:
        raise SystemExit("--dataset-command must write --execution-closure")
```
Add tests proving overrides cannot omit `--limit`, `--rerun`, or `--execution-closure`.

### WR-02: Trust-summary sidecar runs even when dataset closure is absent

**File:** `scripts/release_candidate_validation.py:198`
**Classification:** WARNING
**Issue:** `_dataset_slice_results()` always runs `report_trust_summary.py` immediately after `run_dataset.py`. If the dataset command fails or does not create `execution_closure.json`, `report_trust_summary.py` fails while reading the missing closure, producing a second noisy diagnostic failure instead of recording that the trust summary was skipped because its input sidecar is unavailable. This conflicts with the plan's "when their input artifacts exist" behavior and makes release result review less precise.
**Fix:** Execute the dataset command first, then gate trust-summary generation on `closure_path.exists()` and the dataset result status:
```python
dataset_result = _run_check(...)
results = [dataset_result]
if dataset_result.status == "passed" and closure_path.exists():
    results.append(_run_check(name="trust_summary", ...))
else:
    results.append(
        _skipped_result(
            name="trust_summary",
            command=trust_command,
            classification="diagnostic-only",
            next_action="Generate trust summary after execution_closure.json exists.",
        )
    )
return results
```
Add a test where the dataset command fails and assert the trust-summary row is skipped rather than executed.

---

_Reviewed: 2026-06-01T10:14:24Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
