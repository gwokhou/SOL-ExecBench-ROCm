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

- **After every task commit:** Run the task-specific `verify` command in the plan file.
- **After every plan wave:** Run the wave-level Docker command when Docker/ROCm access is available.
- **Before `$gsd-verify-work`:** Docker build succeeds and in-container Docker dependency tests pass.
- **Max feedback latency:** 300 seconds for host-side tests; Docker build latency is accepted for phase gate validation.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01-01 | 1 | ENV-01, ENV-02 | T-01-01 | Docker runtime flags avoid over-broad mounts beyond ROCm device access | static/smoke | `grep -v '^#' docker/Dockerfile | grep -q 'rocm/dev-ubuntu-24.04:7.1.1-complete'` | W0 | pending |
| 01-01-02 | 01-01 | 1 | ENV-01, ENV-02 | T-01-01 | Docker run script uses AMD device passthrough flags only | static/smoke | `grep -q -- '--device=/dev/kfd' scripts/run_docker.sh` | W0 | pending |
| 01-01-03 | 01-01 | 1 | ENV-02 | T-01-01 | Entrypoint probes ROCm/HIP availability instead of CUDA-only text | static | `grep -q 'torch.version.hip' docker/entrypoint.sh` | W0 | pending |
| 01-02-01 | 01-02 | 1 | ENV-03, SCFG-03 | T-01-02 | Static dependency test file parses `pyproject.toml` with `tomllib` | unit/collection | `uv run pytest tests/docker/dependencies/test_python_dependencies.py -n 0 --collect-only` | W0 | pending |
| 01-02-02 | 01-02 | 1 | ENV-03, SCFG-03 | T-01-02 | `pyproject.toml` resolves PyTorch/torchvision from ROCm wheel index | unit | `uv run pytest tests/docker/dependencies/test_python_dependencies.py -n 0` | W0 | pending |
| 01-02-03 | 01-02 | 1 | ENV-03, SCFG-03 | T-01-02 | `uv.lock` remains synchronized after dependency migration | integration | `uv lock --check` | W0 | pending |
| 01-03-01 | 01-03 | 2 | ENV-02, ENV-04 | T-01-03 | ROCm runtime smoke tests report missing tooling by command name | integration | `uv run pytest tests/docker/dependencies/test_rocm_runtime.py tests/docker/dependencies/test_hip.py -n 0` | W0 | pending |
| 01-03-02 | 01-03 | 2 | ENV-03, ENV-04 | T-01-04 | PyTorch/Triton/ROCm library smoke tests run with ROCm dependencies installed | integration | `uv run pytest tests/docker/dependencies/test_pytorch_rocm.py tests/docker/dependencies/test_triton_rocm.py tests/docker/dependencies/test_rocm_libraries.py -n 0` | W0 | pending |
| 01-04-01 | 01-04 | 3 | ENV-04, SCFG-03 | T-01-10 | Obsolete CUDA/CUTLASS/CuDNN tests are absent from Docker dependency collection | static | `! find tests/docker/dependencies -maxdepth 1 -type f | grep -E 'test_(cuda|cudnn|cutlass)\.py'` | W0 | pending |
| 01-04-02 | 01-04 | 3 | ENV-04, SCFG-03 | T-01-11 | Duplicate CUDA DSL/cuTile/import-only Triton tests are absent from Docker dependency collection | static | `uv run pytest tests/docker/dependencies -n 0 && ! find tests/docker/dependencies -maxdepth 1 -type f | grep -E 'test_(cudnn_frontend|cutedsl|cutile|triton)\.py|_cutedsl_kernels\.py'` | W0 | pending |

---

## Wave 0 Requirements

- [ ] `tests/docker/dependencies/test_python_dependencies.py` - static source guard for ENV-03 and SCFG-03.
- [ ] `tests/docker/dependencies/test_rocm_runtime.py` - ROCm runtime smoke test for ENV-02 and ENV-04.
- [ ] `tests/docker/dependencies/test_hip.py` - HIP compiler smoke test for ENV-02 and ENV-04.
- [ ] `tests/docker/dependencies/test_pytorch_rocm.py` - PyTorch ROCm smoke test for ENV-03 and ENV-04.
- [ ] `tests/docker/dependencies/test_triton_rocm.py` - Triton ROCm smoke test for ENV-03 and ENV-04.
- [ ] `tests/docker/dependencies/test_rocm_libraries.py` - selected ROCm library checks for ENV-02 and ENV-04.

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
