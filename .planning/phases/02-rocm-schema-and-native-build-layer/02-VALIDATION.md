---
phase: 02
slug: rocm-schema-and-native-build-layer
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-21
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/core/data/test_solution.py tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/driver/test_build_ext.py tests/sol_execbench/test_rocm_schema_build_audit.py` |
| **Full suite command** | `uv run pytest tests/sol_execbench/core/data/test_solution.py tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/driver/test_build_ext.py tests/sol_execbench/test_rocm_schema_build_audit.py` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run the quick run command.
- **After every plan wave:** Run the full suite command.
- **Before `$gsd-verify-work`:** Full suite must be green.
- **Max feedback latency:** 30 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | SCFG-01, SCFG-02 | T-02-01 | Reject unsupported CUDA/NVIDIA language values with explicit errors | unit | `uv run pytest tests/sol_execbench/core/data/test_solution.py` | ✅ W0 | ⬜ pending |
| 02-02-01 | 02 | 1 | BUILD-02 | T-02-02 | Inject only ROCm `--offload-arch` flags, no shell interpolation | unit | `uv run pytest tests/sol_execbench/driver/test_problem_packager.py` | ✅ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | BUILD-01, BUILD-03 | T-02-03 | Preserve `benchmark_kernel.so` contract while rejecting CUDA source suffixes | unit | `uv run pytest tests/sol_execbench/driver/test_build_ext.py` | ✅ W0 | ⬜ pending |
| 02-04-01 | 04 | 2 | BUILD-04 | T-02-04 | Fail CUDA/NVIDIA residue in Phase 2-owned schema/build files unless allowlisted with reason | static | `uv run pytest tests/sol_execbench/test_rocm_schema_build_audit.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/sol_execbench/test_rocm_schema_build_audit.py` — audit guard for BUILD-04.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all MISSING references.
- [x] No watch-mode flags.
- [x] Feedback latency < 30s.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending
