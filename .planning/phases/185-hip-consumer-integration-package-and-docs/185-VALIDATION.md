---
phase: 185
slug: hip-consumer-integration-package-and-docs
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-16
validated: 2026-06-16
---

# Phase 185 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | pytest + ruff |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest tests/sol_execbench/test_agent_feedback_fixtures.py tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_cli_environment_snapshot.py` |
| Full suite command | `uv run pytest tests/sol_execbench/test_agent_feedback_fixtures.py tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_v1_20_evidence_quality_docs.py tests/sol_execbench/test_claim_upgrade.py` |
| Estimated runtime | ~4 seconds |

## Sampling Rate

- After every task commit: run the quick command.
- After every plan wave: run the full suite command.
- Before milestone audit: run ruff on changed Python files.
- Max feedback latency: under 10 seconds.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 185-01-01 | 185-01 | 1 | FIXT-01 | — | Fixtures cover valid, missing/unavailable, malformed, stale, partial, and contradictory-authority sidecar cases. | unit/fixture | `uv run pytest tests/sol_execbench/test_agent_feedback_fixtures.py` | yes | green |
| 185-01-02 | 185-01 | 1 | FIXT-02 | — | Docs explain HIP mapping and safe unknown-value downgrade behavior. | docs/unit | `uv run pytest tests/sol_execbench/test_agent_feedback_fixtures.py` | yes | green |
| 185-01-03 | 185-01 | 1 | FIXT-03 | — | Fixture tests prove deterministic, prompt-safe content with no raw profiler dumps, full source, raw trace rows, or absolute temp paths. | unit/fixture | `uv run pytest tests/sol_execbench/test_agent_feedback_fixtures.py` | yes | green |

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
