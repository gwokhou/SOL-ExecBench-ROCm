# Phase 01: ROCm Environment Baseline - Pattern Map

**Created:** 2026-05-21
**Status:** Ready for planning

## Files to Modify

| File | Role | Closest Analog | Notes |
|------|------|----------------|-------|
| `docker/Dockerfile` | ROCm container baseline | Current `docker/Dockerfile` | Replace `nvidia/cuda` base, CUDA env vars, NVIDIA CUTLASS install, and CUDA-oriented sudo/entrypoint assumptions with ROCm equivalents. |
| `scripts/run_docker.sh` | Container launch wrapper | Current `scripts/run_docker.sh` | Replace `--gpus all` with AMD device passthrough flags and keep the existing build/run UX. |
| `docker/entrypoint.sh` | Container startup hook | Current `docker/entrypoint.sh` | Update startup probe so it does not require CUDA-only device naming. |
| `pyproject.toml` | Python dependency source of truth | Current `pyproject.toml` | Replace CUDA wheel index and NVIDIA-only packages with ROCm-compatible dependency declarations. |
| `uv.lock` | Locked dependency resolution | Existing lockfile | Refresh after `pyproject.toml` changes so the lock matches the ROCm wheel source. |
| `tests/docker/dependencies/test_cuda.py` | CUDA dependency smoke test | Current Docker dependency test | Replace with ROCm runtime/toolchain/library smoke checks or remove if superseded. |
| `tests/docker/dependencies/test_triton.py` | Triton import smoke test | Current Triton dependency test | Update to assert ROCm-compatible Triton import/runtime behavior. |
| `tests/docker/dependencies/*.py` | Container dependency coverage | Docker dependency suite | Add ROCm runtime, HIP compiler, PyTorch ROCm, and selected ROCm library checks. |

## Codebase Patterns

### Dockerfile Pattern
- `docker/Dockerfile` uses a pinned base image, copies `pyproject.toml`/`uv.lock` early, installs dependencies with `uv sync`, then copies `src/` and installs the project itself.
- Keep that staged layering so image rebuilds remain cache-friendly.

### Docker Runtime Pattern
- `scripts/run_docker.sh` owns build and run flags, mounts the repository, and defaults to an interactive shell.
- Preserve the script’s interface; only swap NVIDIA runtime flags for ROCm device passthrough.

### Test Pattern
- `tests/docker/dependencies/` uses small subprocess-based smoke tests with explicit assertions and readable failure messages.
- Preserve that style for ROCm checks so missing tools are easy to distinguish from broken installs.

### Packaging Pattern
- `pyproject.toml` owns uv sources and dependency declarations.
- Update the dependency source there first, then refresh the lockfile.

## Source-of-Truth References

- `docker/Dockerfile`
- `scripts/run_docker.sh`
- `docker/entrypoint.sh`
- `pyproject.toml`
- `tests/docker/dependencies/test_cuda.py`
- `tests/docker/dependencies/test_triton.py`

