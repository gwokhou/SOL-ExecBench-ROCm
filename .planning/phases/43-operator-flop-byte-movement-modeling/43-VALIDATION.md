---
phase: 43
slug: operator-flop-byte-movement-modeling
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-23
---

# Phase 43 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` |
| **Estimated runtime** | ~30 seconds for focused CPU tests |

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x` once the file exists; for extractor-only work, run `uv run pytest tests/sol_execbench/test_amd_bound_graph.py -x`.
- **After every plan wave:** Run the full suite command above.
- **Before `$gsd-verify-work`:** Full suite plus Ruff check must be green.
- **Max feedback latency:** 60 seconds for focused CPU tests.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 43-01-01 | 01 | 1 | MODEL-05 | T-43-01 | Derived estimates do not mutate public schemas | unit | `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x` | W0 | pending |
| 43-01-02 | 01 | 1 | MODEL-05 | T-43-01 | Public exports are deliberate | unit | `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x` | W0 | pending |
| 43-02-01 | 02 | 2 | MODEL-01 | T-43-02 | GEMM evidence is shape-derived and auditable | unit | `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x` | W0 | pending |
| 43-02-02 | 02 | 2 | MODEL-02 | T-43-02 | Elementwise/activation chains are not hidden | unit | `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x` | W0 | pending |
| 43-03-01 | 03 | 2 | MODEL-03 | T-43-03 | Axis gaps degrade to inexact evidence | unit | `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x` | W0 | pending |
| 43-03-02 | 03 | 2 | MODEL-04 | T-43-03 | Logical views and materialization are distinguishable | unit | `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x` | W0 | pending |
| 43-04-01 | 04 | 3 | MODEL-05 | T-43-04 | Legacy adapters preserve visible unsupported debt | regression | `uv run pytest tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` | W0 | pending |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification.

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-23
