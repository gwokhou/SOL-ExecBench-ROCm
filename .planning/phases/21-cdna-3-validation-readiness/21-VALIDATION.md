---
phase: 21
slug: cdna-3-validation-readiness
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-22
---

# Phase 21 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py` |
| **Estimated runtime** | ~15 seconds |

## Per-Task Verification Map

| Task ID | Requirement | Test Type | Automated Command | Status |
|---------|-------------|-----------|-------------------|--------|
| 21-01 | VAL-04 | unit | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py` | pending |
| 21-02 | VAL-05 | docs/unit | `uv run pytest tests/sol_execbench/test_rocm_support_docs.py` | pending |
| 21-03 | VAL-06 | unit | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_public_contract_guardrails.py` | pending |

## Manual-Only Verifications

Real CDNA 3 hardware validation is intentionally manual/future scope and is not
required for Phase 21.

## Validation Sign-Off

- [x] Unit tests cover readiness behavior without real CDNA 3 hardware.
- [x] Docs distinguish readiness from validation pass.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-22
