---
phase: 79
slug: docker-matrix-selection-and-preflight
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-28
---

# Phase 79 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Pytest 9.0.2 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` |
| **Full suite command** | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py -q` |
| **Estimated runtime** | ~5 seconds for focused CPU-safe tests |

---

## Sampling Rate

- **After every task commit:** Run the plan's listed focused pytest command.
- **After every plan wave:** Run the full Phase 79 suite listed above.
- **Before `$gsd-verify-work`:** Full suite and Ruff must be green.
- **Max feedback latency:** 10 seconds for CPU-safe checks.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 79-01-01 | 79-01 | 1 | DOCKER-01, DOCKER-03, DOCKER-05 | T-79-01, T-79-02 | Unknown Targets rejected unless explicit non-authoritative override is supplied | unit/static | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_docker_matrix_targets.py -q` | ❌ W0 | ⬜ pending |
| 79-01-02 | 79-01 | 1 | DOCKER-04, DOCKER-05 | T-79-03, T-79-04 | Docker runtime failures classify as `runtime_unavailable` before benchmark execution | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_docker_matrix_preflight.py -q` | ❌ W0 | ⬜ pending |
| 79-01-03 | 79-01 | 1 | DOCKER-03, DOCKER-05 | T-79-02, T-79-04 | JSON preview exposes build args and authority-false decision fields without live Docker | unit/subprocess | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` | ❌ W0 | ⬜ pending |
| 79-02-01 | 79-02 | 2 | DOCKER-02 | T-79-07 | Dockerfile keeps the ROCm 7.1 default while accepting image/tag args | static | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_docker_matrix_script.py -q` | ❌ W0 | ⬜ pending |
| 79-02-02 | 79-02 | 2 | DOCKER-03, DOCKER-05 | T-79-06, T-79-07 | Script resolves declared Targets and passes selected build args without broad privileges | static/subprocess | `bash -n scripts/run_docker.sh && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_docker_matrix_script.py -q` | ❌ W0 | ⬜ pending |
| 79-02-03 | 79-02 | 2 | DOCKER-04, DOCKER-05 | T-79-08, T-79-09 | Preflight-only and runtime-unavailable paths stop before Docker build/run and benchmark commands | static/subprocess | `bash -n scripts/run_docker.sh && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_docker_matrix_script.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/sol_execbench/test_docker_matrix_targets.py` — stubs for DOCKER-01, DOCKER-02, DOCKER-03, DOCKER-05.
- [ ] `tests/sol_execbench/test_docker_matrix_preflight.py` — stubs for DOCKER-03, DOCKER-04, DOCKER-05.
- [ ] `tests/sol_execbench/test_run_docker_matrix_script.py` — stubs for DOCKER-02, DOCKER-03, DOCKER-04, DOCKER-05.
- [ ] `src/sol_execbench/core/docker_matrix.py` — pure helper surface used by CPU-safe tests and script integration.
- [ ] `docker/rocm-targets.json` — checked-in manifest consumed by the helper.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live ROCm container benchmark execution | DOCKER-04 | Current local environment is missing `/dev/dri`; Phase 79 is required to be CPU-safe and should classify this as `runtime_unavailable` rather than require live execution. | Optional after hardware is available: run `./scripts/run_docker.sh --preflight-only --target <declared-target>` and confirm it reports runtime availability before attempting a benchmark. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 10s for CPU-safe focused checks.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-28
