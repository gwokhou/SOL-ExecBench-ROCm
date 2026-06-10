---
phase: 175
slug: pid-lock-module
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-10
---

# Phase 175 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2+ |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/sol_execbench/core/bench/test_pid_lock.py -x` |
| **Full suite command** | `uv run pytest tests/` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/sol_execbench/core/bench/test_pid_lock.py -x`
- **After every plan wave:** Run `uv run pytest tests/sol_execbench/core/bench/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 175-01-01 | 01 | 1 | INST-01 | T-175-01 / — | Path validation via pathlib.Path | unit | `uv run pytest tests/sol_execbench/core/bench/test_pid_lock.py::test_exclusive_acquire -x` | ❌ W0 | ⬜ pending |
| 175-01-02 | 01 | 1 | INST-01 | — | N/A | integration | `uv run pytest tests/sol_execbench/core/bench/test_pid_lock.py::test_contention_exits_with_diagnostic -x` | ❌ W0 | ⬜ pending |
| 175-02-01 | 02 | 1 | INST-02 | — | N/A | integration | `uv run pytest tests/sol_execbench/core/bench/test_pid_lock.py::test_auto_release_on_sigkill -x` | ❌ W0 | ⬜ pending |
| 175-02-02 | 02 | 1 | INST-02 | — | N/A | integration | `uv run pytest tests/sol_execbench/core/bench/test_pid_lock.py::test_auto_release_on_normal_exit -x` | ❌ W0 | ⬜ pending |
| 175-03-01 | 03 | 2 | INST-03 | — | N/A | integration | `uv run pytest tests/sol_execbench/core/bench/test_pid_lock.py::test_timing_batch_mandatory_lock -x` | ❌ W0 | ⬜ pending |
| 175-03-02 | 03 | 2 | INST-03 | — | N/A | integration | `uv run pytest tests/sol_execbench/core/bench/test_pid_lock.py::test_overhead_calibration_mandatory_lock -x` | ❌ W0 | ⬜ pending |
| 175-03-03 | 03 | 2 | INST-03 | — | N/A | integration | `uv run pytest tests/sol_execbench/core/bench/test_pid_lock.py::test_derived_isolated_optional_lock -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/sol_execbench/core/bench/test_pid_lock.py` — stubs for INST-01, INST-02, INST-03
- [ ] `tests/conftest.py` — existing fixtures (`tmp_cache_dir`, `tmp_path`) sufficient

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | — | — | — |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
