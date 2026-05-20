---
phase: 01
slug: rocm-environment-baseline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-21
---

# Phase 01 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest with pytest-xdist |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/docker/dependencies -n 0` |
| **Full suite command** | `./scripts/run_docker.sh --build -- pytest tests/docker/dependencies -n 0` |
| **Estimated runtime** | ~300 seconds for host-side checks; Docker build/runtime depends on image cache |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/docker/dependencies -n 0` where tests can run in the current environment.
- **After every plan wave:** Run `./scripts/run_docker.sh --build -- pytest tests/docker/dependencies -n 0` when Docker/ROCm device access is available.
- **Before `$gsd-verify-work`:** Docker build succeeds and in-container Docker dependency tests pass.
- **Max feedback latency:** 300 seconds for host-side tests; Docker build latency is accepted for phase gate validation.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | ENV-01, ENV-02 | T-01-01 | Docker runtime flags avoid over-broad mounts beyond ROCm device access | static/smoke | `uv run pytest tests/docker/dependencies -n 0` | W0 | pending |
| 01-02-01 | 02 | 1 | ENV-03, SCFG-03 | T-01-02 | Python dependencies resolve from explicit ROCm wheel sources without CUDA/NVIDIA packages | static | `uv lock --check` and Docker dependency package tests | W0 | pending |
| 01-03-01 | 03 | 1 | ENV-02, ENV-04 | T-01-03 | Dependency tests use fixed command lists and clear failure messages | integration | `uv run pytest tests/docker/dependencies -n 0` | W0 | pending |
| 01-04-01 | 04 | 2 | ENV-01, ENV-02, ENV-03, ENV-04, SCFG-03 | T-01-04 | In-container validation proves ROCm runtime, HIP, PyTorch ROCm, Triton ROCm, and selected ROCm libraries | integration | `./scripts/run_docker.sh --build -- pytest tests/docker/dependencies -n 0` | W0 | pending |

---

## Wave 0 Requirements

- [ ] `tests/docker/dependencies/test_rocm_runtime.py` - stubs/checks for ENV-02 and ENV-04.
- [ ] `tests/docker/dependencies/test_hip.py` - HIP compiler smoke test for ENV-02 and ENV-04.
- [ ] `tests/docker/dependencies/test_pytorch_rocm.py` - PyTorch ROCm backend smoke test for ENV-03 and ENV-04.
- [ ] `tests/docker/dependencies/test_triton_rocm.py` - Triton ROCm import/minimal runtime smoke test for ENV-03 and ENV-04.
- [ ] `tests/docker/dependencies/test_rocm_libraries.py` - selected ROCm library discovery/link checks for ENV-02 and ENV-04.
- [ ] Static dependency assertion covering absence of CUDA/NVIDIA package declarations for SCFG-03.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Docker build and ROCm GPU passthrough on AMD host | ENV-01, ENV-02, ENV-03, ENV-04 | Requires Docker access and AMD GPU device nodes on the host | Run `./scripts/run_docker.sh --build -- pytest tests/docker/dependencies -n 0` on a ROCm host and inspect failures for missing tooling vs test failures. |

---

## Validation Sign-Off

- [ ] All tasks have automated verify commands or Wave 0 dependencies.
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify.
- [ ] Wave 0 covers all missing Docker dependency test references.
- [ ] No watch-mode flags.
- [ ] Host-side feedback latency < 300s where available.
- [ ] Docker build/runtime phase gate recorded.
- [ ] `nyquist_compliant: true` set in frontmatter when execution proves coverage.

**Approval:** pending
