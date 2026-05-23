---
phase: 44
slug: bound-artifact-v2-and-coverage-semantics
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-23
---

# Phase 44 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` |
| **Estimated runtime** | ~30 seconds for focused CPU tests |

## Sampling Rate

- **After every task commit:** Run the quick command once
  `tests/sol_execbench/test_amd_sol_v2.py` exists.
- **After every plan wave:** Run the full suite command above.
- **Before `$gsd-verify-work`:** Full suite plus Ruff check must be green.
- **Max feedback latency:** 60 seconds for focused CPU tests.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 44-01-01 | 01 | 1 | BOUND-01 | T-44-01 | Sidecars reject malformed schemas | unit | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` | W0 | pending |
| 44-01-02 | 01 | 1 | BOUND-01 | T-44-01 | Round-trip serialization preserves evidence fields | unit | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` | W0 | pending |
| 44-02-01 | 02 | 2 | BOUND-02 | T-44-02 | Per-op bounds are derived from rich estimates and hardware model limits | unit | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` | W0 | pending |
| 44-02-02 | 02 | 2 | BOUND-04 | T-44-03 | Unsupported evidence creates unscored aggregate state | unit | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` | W0 | pending |
| 44-03-01 | 03 | 2 | BOUND-03 | T-44-04 | Coverage reports counts by family and worst confidence | unit | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` | W0 | pending |
| 44-03-02 | 03 | 2 | BOUND-04 | T-44-04 | Warning strings are deterministic and visible | unit | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x` | W0 | pending |
| 44-04-01 | 04 | 3 | BOUND-01 | T-44-05 | v1 and canonical public contracts remain unchanged | regression | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` | W0 | pending |

## Wave 0 Requirements

Existing Phase 42 and Phase 43 graph/estimate tests cover the inputs consumed
by this phase. No external GPU or ROCm hardware setup is required for Phase 44.

## Manual-Only Verifications

All Phase 44 behaviors have automated verification.

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-23
