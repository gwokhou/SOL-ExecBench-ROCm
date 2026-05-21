---
phase: 03
slug: rocm-evaluation-timing-and-hardware-introspection
status: executed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-21
validated: 2026-05-21
---

# Phase 03 — Validation Strategy

> Retroactive Nyquist validation contract reconstructed from Phase 03 plans,
> summaries, and verification evidence.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --no-sync pytest tests/sol_execbench/driver/test_eval_driver.py tests/sol_execbench/core/bench/test_clock_lock.py tests/sol_execbench/test_rocm_eval_timing_audit.py` |
| **Full suite command** | `uv run --no-sync pytest tests/sol_execbench/driver/test_eval_driver.py tests/sol_execbench/core/bench/test_timing.py tests/sol_execbench/core/bench/test_clock_lock.py tests/sol_execbench/test_rocm_eval_timing_audit.py` |
| **Estimated runtime** | ~15 seconds; timing tests may be marker-skipped without matching GPU test configuration |

---

## Sampling Rate

- **After every task commit:** Run the task-specific pytest target listed in the plan or the quick command above.
- **After every plan wave:** Run the full suite command above.
- **Before `$gsd-verify-work`:** Phase 03 verification command set must be green or explicitly skipped by marker.
- **Max feedback latency:** 60 seconds for focused tests.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 03-01 | 1 | EVAL-01, EVAL-02, EVAL-03 | T-03-01 | Eval driver routes ROCm native categories and emits strict trace JSONL | unit/e2e | `uv run --no-sync pytest tests/sol_execbench/driver/test_eval_driver.py` | ✅ W0 | ✅ green |
| 03-02-01 | 03-02 | 1 | EVAL-04, EVAL-05 | T-03-02 | DPS and return-value conventions preserve input/output semantics | unit/e2e | `uv run --no-sync pytest tests/sol_execbench/driver/test_eval_driver.py` | ✅ W0 | ✅ green |
| 03-03-01 | 03-03 | 2 | PROF-01, PROF-02, PROF-03 | T-03-03 | Timing avoids CUPTI and uses ROCm-compatible HIP-backed device events | unit/static | `uv run --no-sync pytest tests/sol_execbench/core/bench/test_timing.py tests/sol_execbench/test_rocm_eval_timing_audit.py` | ✅ W0 | ✅ green/skipped by marker |
| 03-04-01 | 03-04 | 2 | PROF-04, PROF-05 | T-03-04 | Environment and clock checks use ROCm/HIP and `rocm-smi`, not NVIDIA tooling | unit/static | `uv run --no-sync pytest tests/sol_execbench/core/bench/test_clock_lock.py tests/sol_execbench/test_rocm_eval_timing_audit.py` | ✅ W0 | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/sol_execbench/driver/test_eval_driver.py` — trace emission, runtime statuses, DPS/return conventions, and reward-hack interaction.
- [x] `tests/sol_execbench/core/bench/test_timing.py` — timing API semantics; marker-skipped when the required GPU timing configuration is unavailable.
- [x] `tests/sol_execbench/core/bench/test_clock_lock.py` — ROCm clock lock, unlock, and verification behavior.
- [x] `tests/sol_execbench/test_rocm_eval_timing_audit.py` — static guard for ROCm timing/hardware-introspection paths.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU timing integration on target ROCm hardware | PROF-02, PROF-03 | Requires matching ROCm GPU timing environment; Phase 05 owns hardware matrix execution | Run the full adapted suite on target ROCm hardware and inspect Phase 05 hardware evidence. |

---

## Validation Audit 2026-05-21

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 1 manual-only hardware timing confirmation |

---

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 60s for focused tests.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-21
