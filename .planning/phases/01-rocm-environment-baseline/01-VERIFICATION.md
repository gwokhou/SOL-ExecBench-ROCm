---
phase: 01
slug: rocm-environment-baseline
status: pass
verified_at: 2026-05-21T11:13:44Z
commit: aa4d075-plus-followup
---

# Phase 01 Verification - ROCm Environment Baseline

## Status

`pass`

Phase 1 implementation is complete and the ROCm Docker runtime validation now passes on the local AMD ROCm host. The final verification run built `sol-execbench:latest` from `rocm/dev-ubuntu-24.04:7.1.1-complete`, passed AMD device nodes through to the container, detected HIP backend `7.1.25424` and GPU `AMD Radeon Graphics`, and passed all six Docker dependency tests.

## Automated Verification

| Check | Result | Notes |
|-------|--------|-------|
| Dockerfile ROCm base and no executable CUDA path residue | PASS | `rocm/dev-ubuntu-24.04:7.1.1-complete` present; CUDA base/path probes absent. |
| Docker run flags use AMD device passthrough | PASS | `/dev/kfd`, `/dev/dri`, `--group-add video` present; executable `--gpus`/`--privileged` absent. |
| Entrypoint HIP probe and no CUDA-only missing-device text | PASS | `torch.version.hip` present; old `No CUDA device detected` text absent. |
| `uv.lock` synchronized | PASS | `uv lock --check` resolved 70 packages. |
| Static dependency guard | PASS | `python -m pytest -p no:xdist -o addopts='' tests/docker/dependencies/test_python_dependencies.py -q`. |
| Docker dependency collection | PASS | 6 ROCm-focused tests collected. |
| Obsolete CUDA/NVIDIA Docker dependency tests absent | PASS | Removed CUDA, cuDNN, CUTLASS, CuTe DSL, cuTile, and old Triton smoke files. |
| HIP compile smoke on host | PASS | `test_hip.py` passed. |
| Selected ROCm library discovery on host | PASS | `test_rocm_libraries.py` passed. |
| ROCm runtime tool smoke on host | PASS | `rocminfo` reports `gfx1200`; `rocm-smi` sees the AMD GPU. |
| Docker build user setup | PASS AFTER FIX | ROCm base image already contains UID 1000; Dockerfile now reuses existing UID/GID users instead of unconditionally running `useradd -u 1000`. |
| Docker build dependency install | PASS | `uv sync --frozen --no-install-project --all-groups` installed `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, and `triton-rocm==3.6.0`. |
| Container device passthrough | PASS | `docker run` used `/dev/kfd`, `/dev/dri`, `--group-add video`, and the entrypoint detected HIP/GPU availability. |
| Docker dependency runtime suite | PASS | `./scripts/run_docker.sh -- pytest tests/docker/dependencies -n 0` passed 6/6 tests. |

## Runtime Verification Completed

Verified on the local ROCm host with AMD GPU device nodes visible:

```bash
./scripts/run_docker.sh -- pytest tests/docker/dependencies -n 0
```

Observed outcome:

- Docker image built from `rocm/dev-ubuntu-24.04:7.1.1-complete`.
- Container saw `/dev/kfd` and `/dev/dri`.
- Entrypoint detected HIP backend `7.1.25424` and GPU `AMD Radeon Graphics`.
- All six tests under `tests/docker/dependencies/` passed in the container.
- Earlier failure modes now have actionable diagnostics: Docker Desktop context is rejected before run, non-TTY runs omit `-it`, and ROCm test assertions match the installed ROCm tools.

## Residual Risk

Clock locking remains best-effort for Phase 1; the container reported `CLOCKS_LOCKED=0` because no GPU clock preset was configured for `AMD Radeon Graphics`. Phase 3 owns deeper ROCm timing and clock-management semantics.

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| ENV-01 | COVERED | ROCm Docker base image and build path updated. |
| ENV-02 | COVERED | ROCm tooling tests pass in the container with AMD device passthrough. |
| ENV-03 | COVERED | ROCm Python dependencies install in Docker; PyTorch ROCm and Triton ROCm smoke tests pass. |
| ENV-04 | COVERED | Docker dependency tests replaced with ROCm-focused collection. |
| SCFG-03 | COVERED | CUDA/NVIDIA-only dependency declarations and smoke tests removed. |
