---
phase: 23
slug: timing-semantics-and-policy
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-22
---

# Phase 23 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py`
- **After every plan wave:** Run `uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py`
- **Before `$gsd-verify-work`:** Full phase command must pass
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 23-01-01 | 01 | 1 | TIME-01 | — | N/A | unit | `uv run pytest tests/sol_execbench/test_timing_policy.py` | ❌ W0 | ⬜ pending |
| 23-01-02 | 01 | 1 | TIME-02 | — | N/A | unit | `uv run pytest tests/sol_execbench/test_timing_policy.py` | ❌ W0 | ⬜ pending |
| 23-01-03 | 01 | 1 | TIME-03 | — | N/A | unit/docs | `uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py` | ❌ W0 | ⬜ pending |
| 23-01-04 | 01 | 1 | TIME-04 | — | N/A | docs | `uv run pytest tests/sol_execbench/test_rocm_eval_timing_audit.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/sol_execbench/test_timing_policy.py` — unit coverage for TIME-01, TIME-02, TIME-03.
- [ ] `docs/rocm_timing.md` — documentation target for TIME-03 and TIME-04.
- [ ] `tests/sol_execbench/test_rocm_eval_timing_audit.py` — extend audit coverage for timing documentation and public-contract guardrails.

---

## Manual-Only Verifications

All Phase 23 behaviors have automated verification. Manual review should still
confirm the timing semantics are understandable because the source-specific
chimney model is user-facing benchmark methodology.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-22
