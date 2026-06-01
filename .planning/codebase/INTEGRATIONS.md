# External Integrations

**Analysis Date:** 2026-06-01

## APIs & External Services

**Dataset Acquisition:**
- Hugging Face Hub dataset `nvidia/SOL-ExecBench` - Public benchmark dataset download and local layout generation in `scripts/download_solexecbench.py`.
  - SDK/Client: `datasets.load_dataset` from the `datasets` package declared in `pyproject.toml`.
  - Auth: Not required by repository code; optional Hugging Face environment/token behavior is inherited from the `datasets` library, not managed in this repo.

**Package Indexes:**
- PyPI - Default Python package index configured as `pypi` in `pyproject.toml`.
  - SDK/Client: uv/pip resolver from `uv.lock`, `pyproject.toml`, and `docker/Dockerfile`.
  - Auth: None configured.
- PyTorch ROCm wheel indexes - ROCm-specific torch, torchvision, and triton-rocm wheels configured in `pyproject.toml`, `docker/Dockerfile`, and `docker/rocm-targets.json`.
  - SDK/Client: uv and `uv pip install`.
  - Auth: None configured.

**Container Registries:**
- Docker Hub ROCm images - Base images such as `rocm/dev-ubuntu-24.04:7.1.1-complete` in `docker/Dockerfile` and `docker/rocm-targets.json`.
  - SDK/Client: Docker CLI in `scripts/run_docker.sh`.
  - Auth: Not configured by repository code.
- GitHub Container Registry `ghcr.io/astral-sh/uv:0.5.11` - Source image for copying the uv binary in `docker/Dockerfile`.
  - SDK/Client: Docker build `COPY --from=ghcr.io/astral-sh/uv:0.5.11`.
  - Auth: Not configured by repository code.

**ROCm Runtime Tooling:**
- AMD ROCm command-line tools - Runtime discovery, clock locking, profiling, and static evidence routes use local executables rather than network APIs.
  - SDK/Client: `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, `rocm-smi`, `rocprofv3`, `rocprofv3-avail`, `llvm-objdump`, and `readelf` in `docker/Dockerfile`, `docker/entrypoint.sh`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and `src/sol_execbench/core/bench/static_kernel_evidence.py`.
  - Auth: Local executable permissions; clock control uses passwordless `sudo` for `amd-smi` or `rocm-smi` configured in `docker/Dockerfile`.

## Data Storage

**Databases:**
- Not detected. The repository does not use PostgreSQL, MySQL, Redis, SQLAlchemy, Django ORM, or application SQLite for app state.
- `rocprofv3` can emit `.rocpd`/SQLite-like profiler artifacts, but these are diagnostic files parsed or registered by `src/sol_execbench/core/bench/rocm_profiler.py`, not an application database.

**File Storage:**
- Local filesystem only.
- Benchmark inputs are read from problem directories containing `definition.json`, `workload.jsonl`, solution JSON, Python/HIP/C++ source files, and optional `config.json`; loaded by `src/sol_execbench/cli/main.py` and staged by `src/sol_execbench/driver/problem_packager.py`.
- Downloaded SOL ExecBench dataset files are written under `data/SOL-ExecBench/benchmark/` by `scripts/download_solexecbench.py`.
- FlashInfer safetensors assets are expected under `data/flashinfer-trace` or `FLASHINFER_TRACE_DIR`; `scripts/run_docker.sh`, `docker/entrypoint.sh`, `src/sol_execbench/driver/templates/eval_driver.py`, and `src/sol_execbench/core/bench/io.py` handle the lookup path.
- Trace JSONL, sidecar diagnostics, compatibility matrices, static evidence, profiler sidecars, and report artifacts are written by `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/runtime_evidence.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, and `scripts/run_dataset.py`.

**Caching:**
- uv cache in Docker is configured with `UV_CACHE_DIR=/home/${HOST_USER}/.cache/uv` and BuildKit cache mounts in `docker/Dockerfile`.
- Docker build cache is used implicitly by `docker/Dockerfile`.
- Python bytecode and pytest caches may exist locally but are not application caches.
- No Redis, Memcached, or remote cache integration detected.

## Authentication & Identity

**Auth Provider:**
- Not detected. There is no application login, OAuth provider, session store, API key middleware, or identity provider in `src/sol_execbench/`.
  - Implementation: CLI-only local execution with file inputs and local subprocess/container boundaries.

**Secret Handling:**
- No required `.env` file detected in repository configuration.
- Do not commit Hugging Face tokens, proprietary kernels, credentials, downloaded datasets, or benchmark outputs; this is stated in `AGENTS.md` and reinforced by local file-based dataset workflows.
- Docker and dataset download integrations rely on external tool defaults for credentials if private resources are used; the repository does not read or store credentials directly.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, OpenTelemetry collector, hosted logging, or external error tracking integration detected.

**Logs:**
- CLI human output uses Click/Rich in `src/sol_execbench/cli/main.py`.
- Evaluation subprocess logs, stdout/stderr tails, no-trace diagnostics, and stage-aware messages are written to local sidecar files by `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/diagnostics.py`, and `src/sol_execbench/core/bench/utils.py`.
- Runtime environment snapshots are optional local JSON sidecars controlled by `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` in `src/sol_execbench/cli/main.py` and `src/sol_execbench/core/environment.py`.
- `rocprofv3` profiling evidence is collected as local diagnostic artifacts by `src/sol_execbench/core/bench/rocm_profiler.py`.

## CI/CD & Deployment

**Hosting:**
- Not applicable. The project is a Python CLI/package and local Docker GPU evaluation environment, not a deployed web service.

**CI Pipeline:**
- No GitHub Actions or other CI workflow files detected in the explored repo tree.
- Test and verification commands are local: `uv run pytest tests/`, `uv run --with ruff ruff check .`, `uv run --with ruff ruff format .`, and Docker GPU checks through `scripts/run_docker.sh`.

## Environment Configuration

**Required env vars:**
- None for normal host CLI startup.

**Important optional env vars:**
- `PYTORCH_ALLOC_CONF` - Set for compile/evaluation subprocesses in `src/sol_execbench/cli/main.py`.
- `PYTORCH_ROCM_ARCH` - Native extension architecture override used by staged HIP/C++ builds.
- `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` - Optional environment snapshot sidecars in `src/sol_execbench/cli/main.py`.
- `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, `CUDA_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL` - Device visibility evidence collected in `src/sol_execbench/core/environment.py` and `scripts/run_docker.sh`.
- `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, `SOL_EXECBENCH_MCLK_LEVEL`, `SOL_EXECBENCH_GPU_CLK_MHZ`, and `SOL_EXECBENCH_DRAM_CLK_MHZ` - Clock locking and diagnostics used by `docker/entrypoint.sh`, `scripts/run_docker.sh`, and `src/sol_execbench/core/bench/clock_lock.py`.
- `FLASHINFER_TRACE_DIR` - Local safetensors lookup root used by Docker and evaluation paths.
- `IMAGE_NAME`, `IMAGE_TAG`, `ROCM_DOCKER_IMAGE`, and `ROCM_DOCKER_TAG` - Docker target/image overrides in `scripts/run_docker.sh`.
- `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES`, `SOL_EXECBENCH_ALLOW_UNTESTED_TARGET_SMOKE`, `SOL_EXECBENCH_RECORD_CONTAINER_VALIDATION`, `SOL_EXECBENCH_HOST_PYTHON`, `SOL_EXECBENCH_COMPATIBILITY_ENTRY`, `SOL_EXECBENCH_COMPATIBILITY_MATRIX`, `SOL_EXECBENCH_RUN_DOCKER_DRY_RUN`, `SOL_EXECBENCH_DOCKER_CONTEXT`, `SOL_EXECBENCH_DOCKER_HOST`, `SOL_EXECBENCH_DEV_*`, `SOL_EXECBENCH_DEPENDENCY_*`, and `SOL_EXECBENCH_RUNTIME_*` - Docker wrapper and compatibility evidence controls in `scripts/run_docker.sh`, `src/sol_execbench/core/dependency_matrix.py`, and `src/sol_execbench/core/runtime_evidence.py`.
- `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, `UV_CACHE_DIR`, `UV_LINK_MODE`, `UV_COMPILE_BYTECODE`, `UV_PYTHON_DOWNLOADS`, and `UV_PROJECT_ENVIRONMENT` - Docker image runtime/install settings in `docker/Dockerfile`.

**Secrets location:**
- Not detected. No repo-managed secret store, `.env`, `.npmrc`, `.pypirc`, or cloud credential files were read or required.

## Webhooks & Callbacks

**Incoming:**
- None. No web server routes, webhook handlers, or callback endpoints detected.

**Outgoing:**
- Hugging Face dataset downloads from `scripts/download_solexecbench.py`.
- Package/container downloads through uv, pip, and Docker in `pyproject.toml`, `docker/Dockerfile`, `docker/rocm-targets.json`, and `scripts/run_docker.sh`.
- No application webhooks, HTTP API clients, or outbound service callbacks detected in `src/sol_execbench/`.

---

*Integration audit: 2026-06-01*
