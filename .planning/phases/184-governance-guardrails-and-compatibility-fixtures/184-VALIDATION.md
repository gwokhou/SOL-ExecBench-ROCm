---
phase: 184
slug: governance-guardrails-and-compatibility-fixtures
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-16
validated: 2026-06-16
---

# Phase 184 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | pytest + ruff |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_claim_upgrade.py tests/sol_execbench/test_v1_20_evidence_quality_docs.py` |
| Full suite command | `uv run pytest tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_claim_upgrade.py tests/sol_execbench/test_v1_20_evidence_quality_docs.py` |
| Estimated runtime | ~3 seconds |

## Sampling Rate

- After every task commit: run the quick command.
- After every plan wave: run the full suite command.
- Before milestone audit: run ruff on changed Python files.
- Max feedback latency: under 10 seconds.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 184-01-01 | 184-01 | 1 | GOVR-01 | — | Authority schema enforces diagnostic-only and false benchmark/release authority flags. | unit | `uv run pytest tests/sol_execbench/test_agent_feedback.py` | yes | green |
| 184-01-02 | 184-01 | 1 | GOVR-02 | — | Contradictory, stale, missing, and parse-failure feedback cannot promote claim-upgrade, release, or cutover authority. | unit | `uv run pytest tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_claim_upgrade.py` | yes | green |
| 184-01-03 | 184-01 | 1 | GOVR-03 | — | Public docs keep feedback sidecars as next-experiment guidance only. | docs/unit | `uv run pytest tests/sol_execbench/test_v1_20_evidence_quality_docs.py` | yes | green |

## Wave 0 Requirements

Existing pytest infrastructure covers all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity has no three consecutive tasks without automated verification.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency is under 10 seconds.
- [x] `nyquist_compliant: true` set in frontmatter.

Approval: approved 2026-06-16.
