---
phase: 19
slug: compatibility-and-practice-inventory
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-22
---

# Phase 19 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_hip_execbench_practice_map.py` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_hip_execbench_practice_map.py` |
| **Estimated runtime** | ~10 seconds |

## Sampling Rate

- **After every task commit:** Run the quick pytest command above.
- **After every plan wave:** Run the full suite command above.
- **Before `$gsd-verify-work`:** Full suite must be green.
- **Max feedback latency:** 30 seconds for Phase 19 targeted tests.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 19-01-01 | 01 | 1 | COMPAT-01 | T-19-01 | Documents current contracts without widening public input/output schemas. | unit | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py` | yes | pending |
| 19-01-02 | 01 | 1 | COMPAT-02 | T-19-02 | Classifies hip-execbench practices without importing incompatible contracts. | unit | `uv run pytest tests/sol_execbench/test_hip_execbench_practice_map.py` | yes | pending |
| 19-01-03 | 01 | 1 | COMPAT-03 | T-19-03 | Detects accidental CLI/API drift from this inventory phase. | unit | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py` | yes | pending |

## Wave 0 Requirements

Existing infrastructure covers all Phase 19 requirements.

## Manual-Only Verifications

All Phase 19 behaviors have automated verification.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency target documented.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-22
