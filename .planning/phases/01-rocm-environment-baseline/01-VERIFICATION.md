---
phase: 01
slug: rocm-environment-baseline
status: human_needed
verified_at: 2026-05-21T05:36:48Z
commit: aa4d075
---

# Phase 01 Verification - ROCm Environment Baseline

## Status

`human_needed`

Phase 1 implementation is complete and static/host-available checks pass. Full Docker runtime validation requires an AMD ROCm environment with `/dev/kfd` and `/dev/dri` visible.

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
| ROCm runtime tool smoke on host | ENV-BLOCKED | `rocminfo` reports `/dev/kfd` missing in the current shell environment. |

## Human Verification Required

Run this on a ROCm host/container setup with AMD GPU device nodes visible:

```bash
./scripts/run_docker.sh --build -- pytest tests/docker/dependencies -n 0
```

Expected outcome:

- Docker image builds from `rocm/dev-ubuntu-24.04:7.1.1-complete`.
- Container sees `/dev/kfd` and `/dev/dri`.
- All six tests under `tests/docker/dependencies/` pass in the container.
- Failures, if any, identify missing ROCm tools or libraries by name.

## Residual Risk

PyTorch/Triton ROCm runtime checks were not executed locally because syncing the ROCm wheel set downloads multi-gigabyte packages and the current shell lacks `/dev/kfd`. The lockfile is resolved, and the tests collect; final runtime proof belongs to the Docker command above.

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| ENV-01 | COVERED | ROCm Docker base image and build path updated. |
| ENV-02 | COVERED / HUMAN RUNTIME NEEDED | ROCm tooling tests exist; host lacks `/dev/kfd` for full runtime proof. |
| ENV-03 | COVERED / HUMAN RUNTIME NEEDED | ROCm Python dependencies resolve; PyTorch/Triton runtime proof needs synced ROCm environment. |
| ENV-04 | COVERED | Docker dependency tests replaced with ROCm-focused collection. |
| SCFG-03 | COVERED | CUDA/NVIDIA-only dependency declarations and smoke tests removed. |

