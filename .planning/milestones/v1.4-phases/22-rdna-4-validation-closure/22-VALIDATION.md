---
phase: 22
slug: rdna-4-validation-closure
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-22
---

# Phase 22 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + `sol-execbench` CLI |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_rocm_support_docs.py` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_e2e.py` |
| **E2E command** | `uv run sol-execbench examples/pytorch/linear_backward --solution examples/pytorch/linear_backward/solution_python.json --output .planning/phases/22-rdna-4-validation-closure/rdna4-linear-backward-traces.jsonl --json` |

## Per-Task Verification Map

| Task ID | Requirement | Test Type | Automated Command | Status |
|---------|-------------|-----------|-------------------|--------|
| 22-01 | RDNA-01 | unit | focused v1.4 unit command | passed |
| 22-02 | RDNA-02 | e2e | `sol-execbench` CLI command | passed |
| 22-03 | RDNA-03 | regression | `tests/sol_execbench/test_e2e.py` plus guardrails | passed |

## Manual-Only Verifications

CDNA 3 hardware validation remains future/manual scope and is not part of Phase
22.

## Validation Sign-Off

- [x] RDNA 4 environment observed as `gfx1200`.
- [x] Focused v1.4 unit tests passed.
- [x] Existing E2E pytest passed.
- [x] `sol-execbench` CLI produced valid trace JSONL with all workloads passed.
- [x] CDNA 3 claim remains deferred.
