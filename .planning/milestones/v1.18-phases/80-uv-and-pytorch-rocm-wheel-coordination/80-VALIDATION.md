---
phase: 80
slug: uv-and-pytorch-rocm-wheel-coordination
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-28
---

# Phase 80 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest `>=9.0.2` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py -q` |
| **Full suite command** | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_dependency_matrix_cli.py tests/sol_execbench/test_run_docker_dependency_preflight.py -q` |
| **Estimated runtime** | ~5 seconds CPU-safe focused suite |

---

## Sampling Rate

- **After every task commit:** Run the narrow dependency policy/classification test for the touched helper plus `bash -n scripts/run_docker.sh` for shell changes.
- **After every plan wave:** Run the full suite command listed above.
- **Before `$gsd-verify-work`:** Full suite and Ruff must be green.
- **Max feedback latency:** 10 seconds for CPU-safe focused checks.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 80-01-01 | 01 | 1 | DEPS-01, DEPS-02 | T-80-01 | Dependency policy remains Target-adjacent and preserves default ROCm 7.1 path | unit/static | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_dependency_matrix_policy.py -q` | ❌ W0 | ⬜ pending |
| 80-01-02 | 01 | 1 | DEPS-03, DEPS-04, DEPS-05, DEPS-07 | T-80-02 | Missing wheels and mismatched installed stacks classify diagnostically with authority false | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_dependency_matrix_classification.py -q` | ❌ W0 | ⬜ pending |
| 80-01-03 | 01 | 1 | DEPS-01, DEPS-03, DEPS-05 | T-80-03 | CLI JSON exposes dependency policy/status without importing live ROCm hardware | unit/subprocess | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_dependency_matrix_cli.py -q` | ❌ W0 | ⬜ pending |
| 80-02-01 | 02 | 2 | DEPS-06, DEPS-07 | T-80-04 | Docker wrapper blocks illegal dependency states before build/run unless explicit debug override is used | script subprocess | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_docker_dependency_preflight.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/sol_execbench/test_dependency_matrix_policy.py` — stubs for DEPS-01 and DEPS-02.
- [ ] `tests/sol_execbench/test_dependency_matrix_classification.py` — stubs for DEPS-03, DEPS-04, DEPS-05, and DEPS-07.
- [ ] `tests/sol_execbench/test_dependency_matrix_cli.py` — stubs for shell-consumable JSON output.
- [ ] `tests/sol_execbench/test_run_docker_dependency_preflight.py` — stubs for DEPS-06 and script-side DEPS-07.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | n/a | All Phase 80 behaviors are covered by CPU-safe policy, classifier, CLI, and shell dry-run tests. | n/a |

---

## Validation Sign-Off

- [x] All tasks have automated verification or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 10s.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-28
