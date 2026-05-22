---
phase: 20
slug: internal-diagnostics-and-evidence-model
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-22
---

# Phase 20 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py` |
| **Estimated runtime** | ~15 seconds |

## Per-Task Verification Map

| Task ID | Requirement | Test Type | Automated Command | Status |
|---------|-------------|-----------|-------------------|--------|
| 20-01 | ENG-04 | unit | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py` | pending |
| 20-02 | ENG-05 | unit | `uv run pytest tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` | pending |
| 20-03 | ENG-06 | unit | `uv run pytest tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py` | pending |

## Manual-Only Verifications

All Phase 20 behaviors have automated verification.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Existing infrastructure covers all phase requirements.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-22
