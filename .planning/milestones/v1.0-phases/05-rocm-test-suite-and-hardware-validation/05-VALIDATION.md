---
phase: 05
slug: rocm-test-suite-and-hardware-validation
status: partial
nyquist_compliant: false
wave_0_complete: true
created: 2026-05-21
validated: 2026-05-21
---

# Phase 05 — Validation Strategy

> Retroactive Nyquist validation contract reconstructed from Phase 05 plans,
> hardware matrix, and verification evidence.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + ruff |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --no-sync pytest tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/test_rocm_library_examples.py tests/examples/test_examples.py -k consistency` |
| **Full suite command** | `uv run --no-sync pytest tests/` |
| **Estimated runtime** | ~5 minutes on the validated RDNA 4 host |

---

## Sampling Rate

- **After every task commit:** Run the affected focused pytest target.
- **After every plan wave:** Run the quick command above.
- **Before `$gsd-verify-work`:** Run the full adapted suite on available ROCm hardware.
- **Max feedback latency:** 300 seconds for full-suite RDNA 4 validation.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 05-01 | 1 | TEST-01, TEST-02 | T-05-01 | Pytest markers distinguish ROCm unavailable, unsupported AMD architecture, RDNA 4, and CDNA 3 | unit/static | `uv run --no-sync pytest tests/sol_execbench/test_rocm_test_suite_audit.py` | ✅ W0 | ✅ green |
| 05-02-01 | 05-02 | 1 | TEST-03, TEST-06 | T-05-02 | Reward-hack tests and driver/helper tests remain active under ROCm | unit/integration | `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py` | ✅ W0 | ✅ green |
| 05-03-01 | 05-03 | 2 | TEST-03, TEST-04 | T-05-03 | Full adapted suite passes on RDNA 4 PyTorch ROCm environment | integration/hardware | `uv run --no-sync pytest tests/` | ✅ W0 | ✅ green |
| 05-03-02 | 05-03 | 2 | TEST-05 | T-05-04 | Full adapted suite passes on CDNA 3 PyTorch ROCm environment before CDNA 3 support is claimed | hardware/manual | `uv run --no-sync pytest tests/` on `gfx94*` | ✅ W0 | ⬜ deferred |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/conftest.py` — ROCm/RDNA4/CDNA3 marker behavior.
- [x] `tests/sol_execbench/test_rocm_test_suite_audit.py` — suite migration audit.
- [x] `tests/sol_execbench/core/bench/test_reward_hack.py` — reward-hack defense checks.
- [x] `tests/examples/test_examples.py` — example execution and consistency checks.
- [x] `tests/sol_execbench/test_e2e.py` — e2e CLI behavior, with external safetensors inputs skipped when absent.
- [x] `.planning/phases/05-rocm-test-suite-and-hardware-validation/05-HARDWARE-MATRIX.md` — hardware evidence matrix.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full adapted suite on CDNA 3 | TEST-05 | Requires CDNA 3 hardware (`gfx94*`) not present on this machine and explicitly deferred by project decision | On a ROCm >= 7.0 CDNA 3 host, run `uv sync --all-groups`, verify PyTorch ROCm sees `gfx94*`, then run `uv run --no-sync pytest tests/`; update `05-HARDWARE-MATRIX.md` and `05-VERIFICATION.md`. |

---

## Validation Audit 2026-05-21

| Metric | Count |
|--------|-------|
| Gaps found | 1 |
| Resolved | 0 |
| Escalated | 1 manual-only deferred hardware item |

---

## Validation Sign-Off

- [x] All non-deferred tasks have automated verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 300s for RDNA 4 full-suite validation.
- [ ] `nyquist_compliant: true` set in frontmatter.

**Approval:** partial 2026-05-21; TEST-05 deferred.
