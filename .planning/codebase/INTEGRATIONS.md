# External Integrations

**Analysis Date:** 2026-06-01

## APIs & External Services

**Dataset Distribution:**
- Hugging Face Datasets - Downloads the public `nvidia/SOL-ExecBench` benchmark dataset into `data/SOL-ExecBench/benchmark`.
  - SDK/Client: `datasets.load_dataset` from `datasets>=4.8.2` in `scripts/download_solexecbench.py`.
  - Auth: Optional Hugging Face credentials handled by the local Hugging Face/datasets environment; no repository-specific token variable is read in `scripts/download_solexecbench.py`.
- Hugging Face Hub CLI - Downloads `flashinfer-ai/flashinfer-trace` into `data/flashinfer-trace`.
  - SDK/Client: `hf download` invoked by `scripts/download_data.sh`; README usage installs it with `uv run --with "huggingface-hub[cli]" ./scripts/download_data.sh`.
  - Auth: Optional Hugging Face CLI authentication outside this repository; no token is stored or read by repo code.

**Package Indexes:**
- PyPI - Default Python package index.
  - SDK/Client: `uv` using `[[tool.uv.index]] name = "pypi"` in `pyproject.toml`.
  - Auth: None configured in repo.
- PyTorch ROCm wheel indexes - ROCm wheels for `torch`, `torchvision`, and `triton-rocm`.
  - SDK/Client: `uv` via `pyproject.toml`, `uv.lock`, `docker/Dockerfile`, and `docker/rocm-targets.json`.
  - Auth: None configured in repo.
  - URLs: `https://download.pytorch.org/whl/rocm7.0`, `https://download.pytorch.org/whl/rocm7.1`, `https://download.pytorch.org/whl/rocm7.2`, and `https://download.pytorch.org/whl/`.

**Container Registries:**
- AMD ROCm Docker images - Base image for benchmark containers.
  - SDK/Client: Docker in `docker/Dockerfile` and `scripts/run_docker.sh`.
  - Auth: Docker daemon/registry auth if required by local environment; none configured in repo.
  - Image: `rocm/dev-ubuntu-24.04` with tags declared in `docker/rocm-targets.json`.
- GitHub Container Registry - Provides the `uv` binary during Docker builds.
  - SDK/Client: Docker `COPY --from=ghcr.io/astral-sh/uv:0.5.11` in `docker/Dockerfile`.
  - Auth: None configured in repo for the public image.

**ROCm System Tools:**
- ROCm runtime/toolchain commands - Local process integrations for GPU discovery, native compilation, profiling, clock locking, static evidence, and diagnostics.
  - SDK/Client: `subprocess` calls in `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/toolchain.py`, and `src/sol_execbench/driver/problem_packager.py`.
  - Auth: Local OS permissions, Docker device access, and passwordless `sudo` for SMI clock controls configured in `docker/Dockerfile`.
  - Tools: `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, `rocm-smi`, `rocprofv3`, `rocprofv3-avail`, `readelf`, `llvm-objdump`, `roc-objdump`, and `rga`.

**CI Services:**
- GitHub Actions - Runs lint, type checks, and CPU-safe tests.
  - SDK/Client: Workflow `.github/workflows/code-quality.yml`.
  - Auth: GitHub Actions default checkout/setup tokens; no custom repository secrets referenced.

## Data Storage

**Databases:**
- Not detected for application data.
  - Connection: Not applicable.
  - Client: Not applicable.
- Local SQLite-like profiler artifacts may be produced by `rocprofv3` as `.rocpd`, `.db`, `.sqlite`, or `.sqlite3` files and are classified in `src/sol_execbench/core/bench/rocm_profiler.py`; these are diagnostic output artifacts, not an application database.

**File Storage:**
- Local filesystem only for benchmark inputs, staged builds, traces, sidecars, reports, static evidence artifacts, and downloaded datasets.
- Dataset roots: `data/SOL-ExecBench/benchmark` from `scripts/download_solexecbench.py` and `data/flashinfer-trace` from `scripts/download_data.sh`.
- Staging directories: temporary `sol_execbench_*` directories created by `src/sol_execbench/cli/main.py` and managed by `src/sol_execbench/driver/problem_packager.py`.
- Output sidecars: environment snapshots, profile metadata, static kernel evidence, compatibility entries, and matrix reports written by `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/runtime_evidence.py`, and `scripts/run_docker.sh`.

**Caching:**
- `uv` package cache in Docker is configured by `UV_CACHE_DIR` in `docker/Dockerfile`.
- Docker build cache mounts are used for apt and uv layers in `docker/Dockerfile`.
- PyTorch/Triton/native compilation caches may be used by underlying tooling, but the repo treats generated build/cache output as non-source artifacts.

## Authentication & Identity

**Auth Provider:**
- None for the benchmark application.
  - Implementation: CLI-only local execution; no login, sessions, users, OAuth, API keys, or app auth provider in `src/sol_execbench/`.
- Optional external identity exists only through user-local tools:
  - Hugging Face CLI/datasets auth may be configured outside the repo for dataset downloads in `scripts/download_data.sh` and `scripts/download_solexecbench.py`.
  - Docker registry auth may be configured outside the repo for image pulls in `docker/Dockerfile` and `scripts/run_docker.sh`.
  - GitHub Actions uses platform-provided identity for `.github/workflows/code-quality.yml`.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, OpenTelemetry, Datadog, Prometheus, or hosted error tracker integration detected.

**Logs:**
- CLI output uses Rich in `src/sol_execbench/cli/main.py`.
- Evaluation subprocesses emit JSONL traces to stdout and non-JSON library messages to stderr in `src/sol_execbench/driver/templates/eval_driver.py`.
- Optional environment snapshots are written when `SOLEXECBENCH_ENV_SNAPSHOT=1` or `SOLEXECBENCH_ENV_SNAPSHOT_PATH` is set in `src/sol_execbench/cli/main.py`.
- Optional `rocprofv3` profile metadata and artifacts are written by `src/sol_execbench/core/bench/rocm_profiler.py`.
- Optional static kernel evidence sidecars are written by `src/sol_execbench/core/bench/static_kernel_evidence.py` and `src/sol_execbench/cli/main.py`.
- Docker compatibility/preflight evidence is written through `scripts/run_docker.sh`, `src/sol_execbench/core/runtime_evidence.py`, and `src/sol_execbench/core/dependency_matrix.py`.

## CI/CD & Deployment

**Hosting:**
- Not applicable. This repository is a local Python package and Dockerized benchmark runner, not a deployed service.
- Container runtime target is local Docker with AMD ROCm device access configured by `scripts/run_docker.sh`.

**CI Pipeline:**
- GitHub Actions workflow `.github/workflows/code-quality.yml` runs on push and pull request.
- CI matrix covers Python `3.12` and `3.13`.
- CI commands: `uv sync --locked --all-groups`, `uv run ruff check .`, `uv run ty check`, CPU-safe `uv run pytest tests/sol_execbench` with GPU-heavy test ignores, and `uv run pytest tests/examples/test_examples.py -k consistency`.

## Environment Configuration

**Required env vars:**
- None for normal host CLI startup.

**Important optional env vars:**
- `PYTORCH_ALLOC_CONF` - Set by CLI subprocess launchers to `expandable_segments:True` in `src/sol_execbench/cli/main.py`.
- `PYTORCH_ROCM_ARCH` - Set by `src/sol_execbench/driver/templates/build_ext.py` when explicit ROCm target hardware exists and the variable is unset.
- `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` - Enable environment snapshot sidecars in `src/sol_execbench/cli/main.py`.
- `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, `CUDA_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL` - Recorded for visibility/runtime evidence in `src/sol_execbench/core/environment.py` and `scripts/run_docker.sh`.
- `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, and `SOL_EXECBENCH_MCLK_LEVEL` - Clock-lock state/config for `docker/entrypoint.sh` and `src/sol_execbench/core/bench/clock_lock.py`.
- `FLASHINFER_TRACE_DIR` - Additional safetensors blob root used by `src/sol_execbench/driver/templates/eval_driver.py` and mounted by `scripts/run_docker.sh`.
- `SOL_EXECBENCH_COMPATIBILITY_ENTRY` and `SOL_EXECBENCH_COMPATIBILITY_MATRIX` - Compatibility evidence output paths consumed by `scripts/run_docker.sh`.
- `SOL_EXECBENCH_DEPENDENCY_*`, `SOL_EXECBENCH_RUNTIME_*`, `SOL_EXECBENCH_DEV_*`, and `SOL_EXECBENCH_DOCKER_*` - Diagnostic/preflight overrides consumed by `scripts/run_docker.sh`, `src/sol_execbench/core/dependency_matrix.py`, and `src/sol_execbench/core/runtime_evidence.py`.
- `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, `UV_CACHE_DIR`, `UV_LINK_MODE`, `UV_COMPILE_BYTECODE`, `UV_PYTHON_DOWNLOADS`, and `UV_PROJECT_ENVIRONMENT` - Docker build/runtime configuration in `docker/Dockerfile`.

**Secrets location:**
- Not detected in repository.
- `.env*` files were not detected during this scan.
- Do not store Hugging Face tokens, Docker credentials, proprietary kernels, or downloaded datasets in the repo; repository guidance in `AGENTS.md` and `docs/DEVELOPMENT.md` treats these as local/external concerns.

## Webhooks & Callbacks

**Incoming:**
- None. No HTTP server, webhook endpoint, ASGI/WSGI app, or callback route detected.

**Outgoing:**
- Dataset downloads to Hugging Face through `scripts/download_solexecbench.py` and `scripts/download_data.sh`.
- Package downloads from PyPI and PyTorch ROCm indexes through `uv` configured in `pyproject.toml`, `uv.lock`, `docker/Dockerfile`, and `docker/rocm-targets.json`.
- Docker pulls from AMD ROCm image registry and GitHub Container Registry through `docker/Dockerfile` and `scripts/run_docker.sh`.
- GitHub Actions pulls public actions `actions/checkout@v4`, `actions/setup-python@v5`, and `astral-sh/setup-uv@v6` in `.github/workflows/code-quality.yml`.

---

*Integration audit: 2026-06-01*
