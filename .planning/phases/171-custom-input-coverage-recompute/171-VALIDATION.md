---
phase: 171-custom-input-coverage-recompute
slug: 171-custom-input-coverage-recompute
status: draft
validated: 2026-06-09
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-09
---

# Phase 171 — Validation Strategy

## Test Infrastructure

| Property | Value |
| --- | --- |
| Framework | pytest |
| Config file | pyproject.toml |
| Quick run command | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_custom_input_transition_ledger.py -q` |
| Full suite command | Same as quick run (single focused Nyquist test surface) |
| Estimated runtime | ~8s |

## Sampling Rate

- **After every task commit:** run quick command.
- **After every plan wave:** run quick command and JSON assertion check.
- **Before `$gsd-verify-work`:** quick command + checksum assertions.
- **Max feedback latency:** ~10s.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 171-01-01 | 01 | 1 | COV-01/COV-02 | T-171-01 | Transition ledger captures all 55 custom-input blocked transitions deterministically | unit | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_custom_input_transition_ledger.py -q` | ✅ | ✅ |
| 171-01-02 | 01 | 1 | COV-01/COV-02 | T-171-01 | Artifacts refreshed and residual classes remain precise | unit/integration check | `python3 -c 'import json, pathlib; d=json.loads(pathlib.Path("out/rdna4-custom-input-transition-ledger/transition-ledger.json").read_text()); assert d.get("cohort_size") == 55; assert all(r.get("transition") for r in d.get("transitions", [])); print("Ledger OK")'` and `python3 -c 'import json; d=json.load(open("out/rdna4-coverage-current/coverage-summary.json")); assert d.get("problem_denominator") == 235; print(d.get("problem_denominator"))'` | ✅ | ✅ |
| 171-01-02 | 01 | 1 | COV-01/COV-02 | T-171-01 | Coverage updates preserve denominator stability | integration check | same as above | ✅ | ✅ |

## Wave 0 Requirements

- [x] `tests/sol_execbench/test_custom_input_transition_ledger.py` exists and passes.
- [x] `out/rdna4-custom-input-transition-ledger/transition-ledger.json` has `cohort_size: 55` and non-empty `transition` fields.
- [x] `out/rdna4-coverage-current/coverage-summary.json` has `problem_denominator: 235`.

## Manual-Only Verifications

*If none: all phase behaviors have automated verification.*

## Validation Audit 2026-06-09

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 3 |
| Escalated | 0 |

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity maintained.
- [x] Wave 0 covers missing references.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-06-09
