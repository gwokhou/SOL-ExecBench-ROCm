---
phase: 182
slug: diagnostic-sidecar-schema-and-generator
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-16
validated: 2026-06-16
---

# Phase 182 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | pytest + ruff |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_cli_environment_snapshot.py` |
| Full suite command | `uv run pytest tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_contract.py` |
| Estimated runtime | ~3 seconds |

## Sampling Rate

- After every task commit: run the quick command.
- After every plan wave: run the full suite command.
- Before milestone audit: run ruff on changed Python files.
- Max feedback latency: under 10 seconds.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 182-01-01 | 182-01 | 1 | SIDE-01 | — | Strict sidecar model validates bounded status, reason, item, limitation, authority, and citation fields. | unit | `uv run pytest tests/sol_execbench/test_agent_feedback.py` | yes | green |
| 182-01-02 | 182-02 | 1 | SIDE-02 | — | CLI persists `trace.jsonl.agent-feedback.json` beside canonical trace output. | unit | `uv run pytest tests/sol_execbench/test_cli_environment_snapshot.py` | yes | green |
| 182-01-03 | 182-01 | 1 | SIDE-03 | — | Builder excludes raw trace rows, raw profiler dumps, full source, prompt text, and unstable temp paths. | unit | `uv run pytest tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_cli_environment_snapshot.py` | yes | green |
| 182-01-04 | 182-01 | 1 | SIDE-04 | — | Missing and unavailable optional diagnostics become limitation states instead of execution failures. | unit | `uv run pytest tests/sol_execbench/test_agent_feedback.py` | yes | green |

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
