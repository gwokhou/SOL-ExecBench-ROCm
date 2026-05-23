---
phase: 42
slug: structured-bound-graph-ir
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-23
---

# Phase 42 - Validation Strategy

## Test Infrastructure

| Property | Value |
| --- | --- |
| Framework | pytest 9.x |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py -x` |
| Full suite command | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` |
| Estimated runtime | ~30-90 seconds |

## Sampling Rate

- After every task commit: run the task's listed quick pytest command.
- After every plan wave: run the focused phase suite.
- Before `$gsd-verify-work`: focused phase suite must be green.
- Max feedback latency: 90 seconds for CPU-only unit coverage.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42-01-01 | 01 | 1 | IR-01, IR-02 | T-42-01 | N/A | unit | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py -x` | W0 | pending |
| 42-01-02 | 01 | 1 | IR-02, IR-03 | T-42-02 | N/A | unit | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py -x` | W0 | pending |
| 42-02-01 | 02 | 2 | IR-01, IR-03 | T-42-03 | Reference tracing stays derived and isolated | unit | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py -x` | W0 | pending |
| 42-02-02 | 02 | 2 | IR-02, IR-03 | T-42-04 | Unsupported evidence is visible | unit | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py -x` | W0 | pending |
| 42-03-01 | 03 | 3 | IR-04 | T-42-05 | Public facade remains compatible | unit | `uv run pytest tests/sol_execbench/test_amd_sol_bounds.py -x` | W0 | pending |
| 42-03-02 | 03 | 3 | IR-04 | T-42-06 | Canonical schemas and CLI remain unchanged | contract | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -x` | W0 | pending |

## Wave 0 Requirements

Existing pytest infrastructure covers all Phase 42 requirements. No new
framework install is required.

## Manual-Only Verifications

All Phase 42 behaviors have automated verification.

## Validation Sign-Off

- [x] All tasks have automated verify commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 90 seconds.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending execution

