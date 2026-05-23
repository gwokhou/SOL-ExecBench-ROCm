---
phase: 56
slug: parity-gap-reporting-and-evidence-review
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-23
---

# Phase 56 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_parity_gap_report.py -n 0 -x` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_parity_gap_report.py tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` |

## Per-Task Verification Map

| Task ID | Plan | Requirement | Test Type | Command | Status |
|---------|------|-------------|-----------|---------|--------|
| 56-01-01 | 01 | GAP-01 | unit | `uv run pytest tests/sol_execbench/test_parity_gap_report.py -n 0 -x` | pending |
| 56-01-02 | 01 | GAP-02 | unit | same | pending |
| 56-01-03 | 01 | GAP-03 | unit | same | pending |
| 56-02-01 | 02 | GAP-04 | unit | same | pending |
| 56-02-02 | 02 | GAP-01 | unit | same | pending |
| 56-02-03 | 02 | GAP-05 | CLI fixture | same | pending |
| 56-03-01 | 03 | GAP-01..GAP-05 | guardrail | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -n 0 -x` | pending |
| 56-03-02 | 03 | GAP-01..GAP-05 | integration | full suite command | pending |

## Manual-Only Verifications

Real full-suite ROCm validation remains outside Phase 56.

## Validation Sign-Off

- [x] All tasks have automated verify commands
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] No watch-mode flags
- [x] `nyquist_compliant: true` set in frontmatter
