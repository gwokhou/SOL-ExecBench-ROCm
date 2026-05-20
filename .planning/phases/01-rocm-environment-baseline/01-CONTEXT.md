# Phase 1: ROCm Environment Baseline - Context

**Gathered:** 2026-05-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the CUDA/NVIDIA development environment with a reproducible ROCm >= 7.0 baseline. This phase covers the Docker base image, ROCm runtime/tooling availability, Python dependency indexes and NVIDIA-only package replacement/removal, and container dependency tests. It does not port schema enums, native HIP build behavior, evaluation timing, examples, or broad test migration except where a Docker dependency smoke test must change to validate the environment baseline.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
All implementation choices are at the agent's discretion — pure infrastructure phase. Preserve existing package layout, Docker script conventions, and benchmark semantics while making the minimum environment changes needed to establish a ROCm-only baseline.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docker/Dockerfile` is the main environment definition. It currently uses `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04`, exports CUDA paths, installs NVIDIA CUTLASS, grants sudo access to `nvidia-smi`, installs dependencies through `uv sync`, and sets `PYTHONPATH=/sol-execbench/src`.
- `docker/entrypoint.sh` performs runtime GPU logging through PyTorch CUDA APIs and should be reviewed for ROCm-compatible startup behavior.
- `pyproject.toml` declares runtime dependencies and uv sources. It currently pins CUDA-oriented packages and an explicit PyTorch CUDA 13.0 wheel index.
- `tests/docker/dependencies/` contains container-level dependency checks and is the natural place to switch Docker smoke tests from CUDA/NVIDIA stack checks to ROCm, HIP, PyTorch ROCm, Triton ROCm, and selected ROCm library checks.
- `scripts/run_docker.sh` is the user-facing Docker entry point and should remain the primary command for building and entering the GPU evaluation container.

### Established Patterns
- The project uses `uv` for dependency management, keeps Docker dependency installation cacheable, and installs the project into `/venv` before allowing mounted source to override the installed package at runtime.
- Pytest markers and Docker dependency tests are already used to separate environment-sensitive checks from ordinary unit tests.
- The current repository is CUDA-oriented in many source, test, and example paths. Phase 1 should only adjust environment baseline files and environment tests; deeper source-level CUDA API migration belongs to later phases.

### Integration Points
- `pyproject.toml` and `uv.lock` define the Python dependency environment consumed by Docker.
- `docker/Dockerfile` consumes `pyproject.toml`, `uv.lock`, and `README.md`, then copies `src/`.
- `tests/docker/dependencies/` verifies Docker image capabilities and should distinguish missing ROCm tooling from test failures.
- Later phases depend on this phase to provide ROCm tooling and Python packages before schema, driver, timing, examples, and broad test migrations proceed.

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. Use ROCm >= 7.0 as the supported baseline, remove CUDA wheel indexes and NVIDIA-only dependency declarations where no ROCm-compatible package exists, and prefer smoke tests that clearly report missing ROCm tools or libraries.

</specifics>

<deferred>
## Deferred Ideas

- HIP/C++ solution staging and compiler flag migration are deferred to Phase 2.
- ROCm evaluation runtime, timing, profiling, and hardware introspection are deferred to Phase 3.
- Public example and library replacement decisions are deferred to Phase 4.
- Full test suite and RDNA 4/CDNA 3 hardware validation are deferred to Phase 5.

</deferred>
