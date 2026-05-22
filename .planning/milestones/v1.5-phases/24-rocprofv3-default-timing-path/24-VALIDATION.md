---
phase: 24
slug: rocprofv3-default-timing-path
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-22
---

# Phase 24 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_rocm_eval_timing_audit.py` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py` |
| **Estimated runtime** | ~45 seconds |

## Sampling Rate

- **After every task commit:** Run quick command.
- **After every plan wave:** Run full suite command.
- **Before verification:** Full suite and ruff checks must pass.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 24-01-01 | 01 | 1 | PROF-01, PROF-04 | unit | `uv run pytest tests/sol_execbench/test_rocm_profiler.py` | ❌ W0 | ⬜ pending |
| 24-01-02 | 01 | 1 | PROF-02, PROF-03 | unit | `uv run pytest tests/sol_execbench/test_rocm_profiler.py` | ❌ W0 | ⬜ pending |
| 24-01-03 | 01 | 1 | PROF-04 | docs/unit | `uv run pytest tests/sol_execbench/test_rocm_eval_timing_audit.py` | ✅ | ⬜ pending |
| 24-01-04 | 01 | 1 | PROF-01-04 | suite | full suite command | mixed | ⬜ pending |

## Wave 0 Requirements

- [ ] `src/sol_execbench/core/bench/rocm_profiler.py`
- [ ] `tests/sol_execbench/test_rocm_profiler.py`

## Manual-Only Verifications

All Phase 24 behaviors have automated verification. Live `rocprofv3` execution
can be added later as hardware evidence, but Phase 24 is validated through
wrapper construction, parser fixtures, and evidence schema assertions.

## Validation Sign-Off

- [x] All tasks have automated verification
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-22
