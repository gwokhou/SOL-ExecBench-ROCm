---
generated_by: gsd-map-codebase
generated_on: 2026-07-09
last_mapped_commit: cc007cd3af3e5100f7d86f155a40d5e51ffb57e5
focus: tech
---

# Integrations

## External Runtime Interfaces

This project is mostly local/offline. Its primary external integrations are
toolchain, hardware, container, package-index, and optional dataset sources
rather than web services.

## ROCm And AMD Hardware

The evaluator integrates with ROCm through PyTorch HIP APIs, HIP/C++ extension
builds, and optional command-line tools. Important integration points:

- PyTorch ROCm device APIs via the historical `torch.cuda` namespace in
  `src/sol_execbench/driver/templates/eval_driver.py` and benchmark helpers.
- HIP native builds in `src/sol_execbench/driver/templates/build_ext.py`.
- Local AMD gfx target probing in `src/sol_execbench/driver/build_config.py`.
- Runtime environment diagnostics in `src/sol_execbench/core/platform/environment*.py`.
- Toolchain routing in `src/sol_execbench/core/platform/toolchain/`.
- ROCm compatibility matrix modeling in `src/sol_execbench/core/platform/compatibility*.py`.

The test harness detects ROCm device nodes `/dev/kfd` and `/dev/dri` in
`tests/conftest.py` before enabling hardware-sensitive tests.

## ROCm Profiling

Optional profiler evidence integrates with `rocprofv3`. The profile path is
diagnostic only and is deliberately separate from correctness authority:

- CLI option: `--profile rocprofv3` in `src/sol_execbench/cli/main.py`.
- Runtime orchestration: `src/sol_execbench/cli/evaluation/runtime.py` and
  `src/sol_execbench/cli/evaluation/phases.py`.
- Profiler implementation: `src/sol_execbench/core/bench/rocm_profiler/`.
- Timing coverage reports: `src/sol_execbench/core/dataset/profiler_timing_coverage/`.
- Documentation: `docs/user/rocm_timing.md` and `docs/user/rocm_toolchain_routing.md`.

Profiler failures fall back to normal evaluation and record bounded diagnostic
metadata where possible.

## Native ROCm Libraries

The supported native library categories are represented as solution schema
languages and runnable examples:

- `hipblas` with example files in `examples/hipblas/gemm/`.
- `miopen` with example files in `examples/miopen/softmax/`.
- `ck` with example files in `examples/ck/gemm/`.
- `rocwmma` with example files in `examples/rocwmma/gemm/`.

Readiness and public-support boundaries are documented in
`docs/user/rocm_libraries.md`. Native build validation is covered by tests such as
`tests/sol_execbench/core/dataset/test_rocm_library_examples.py` and Docker
dependency tests under `tests/docker/dependencies/`.

## Dataset Sources

The repository does not redistribute restricted benchmark datasets. Local
dataset workflows integrate with external source directories supplied by the
operator:

- SOL ExecBench migration: `sol-execbench dataset migrate-sol` and
  `scripts/download_solexecbench.py`.
- FlashInfer trace migration: `sol-execbench dataset migrate-flashinfer`.
- Dataset layout and manifest helpers in `src/sol_execbench/core/dataset/`.
- Operator batch execution in `scripts/run_dataset.py`.

Dataset policy and provenance are documented in `docs/user/provenance.md`,
`docs/user/compliance.md`, `docs/user/COOKBOOK.md`, and related release docs. Migration
manifests preserve source-boundary metadata and local-only evidence references.

## Safetensors And FlashInfer Trace Assets

Workloads can reference safetensors blobs. Staging support lives in
`src/sol_execbench/driver/staging.py` and is called by
`ProblemPackager._stage_safetensors_inputs()` in
`src/sol_execbench/driver/problem_packager.py`. The evaluator consults the
staging directory first and then `FLASHINFER_TRACE_DIR` in
`src/sol_execbench/driver/templates/eval_driver.py`.

## Docker

Docker support integrates with host ROCm device nodes and image dependency
matrices:

- `docker/Dockerfile` installs and verifies ROCm, Python, and package dependencies.
- `docker/entrypoint.sh` provides container entry behavior.
- `docker/rocm-targets.json` defines build/runtime targets and versions.
- `scripts/run_docker.sh` builds and runs the container with ROCm device
  passthrough and dependency overrides.
- Docker matrix and preflight models live in `src/sol_execbench/core/platform/docker_matrix/`.

## Package Indexes

`pyproject.toml` configures uv indexes:

- `https://pypi.org/simple` as the default index.
- `https://download.pytorch.org/whl/rocm7.1` for PyTorch ROCm wheels.
- `https://download.pytorch.org/whl/` for Triton ROCm wheels.

Linux x86_64 markers route PyTorch, torchvision, and Triton ROCm dependencies
to the explicit ROCm indexes.

## GitHub Actions

CI integration is limited to `.github/workflows/code-quality.yml`. It runs on
GitHub-hosted Ubuntu with Python 3.12 and 3.13, installs via uv, runs Ruff and
Ty, then runs CPU-safe tests. Hardware ROCm tests are skipped in CI unless an
external ROCm-capable environment is provided.

## No Auth, Database, Or Webhook Layer

There is no application database, user authentication provider, webhook
consumer, or hosted API service in the codebase. Persistence is file-based:
trace JSONL, sidecar JSON, migration manifests, generated reports, staging
directories, and local benchmark assets.
