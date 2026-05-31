# External Integrations

**Analysis Date:** 2026-05-31

## APIs & External Services

**Dataset Acquisition:**
- Hugging Face Datasets - Downloads public SOL-ExecBench benchmark data in `scripts/download_solexecbench.py`.
  - SDK/Client: `datasets.load_dataset` from the `datasets>=4.8.2` dependency.
  - Auth: Not required by code; standard Hugging Face environment or CLI auth may be used externally if the dataset access policy requires it.
  - Dataset ID: `nvidia/SOL-ExecBench`.
  - Output: `data/SOL-ExecBench/benchmark/`.
- Hugging Face Hub CLI - Downloads FlashInfer trace data in `scripts/download_data.sh`.
  - SDK/Client: `hf download` supplied by `huggingface-hub[cli]`.
  - Auth: Standard Hugging Face CLI auth if needed; no repository-specific token variable is read by the code.
  - Dataset ID: `flashinfer-ai/flashinfer-trace`.
  - Revision: `1.0`.
  - Output: `data/flashinfer-trace/`.

**Package Indexes:**
- PyPI - Default package index for `uv` dependency resolution in `pyproject.toml`.
  - SDK/Client: `uv`.
  - Auth: None detected.
  - URL: `https://pypi.org/simple`.
- PyTorch ROCm wheel indexes - Explicit `uv` indexes for ROCm Torch, torchvision, and Triton ROCm wheels in `pyproject.toml` and Docker target policies in `docker/rocm-targets.json`.
  - SDK/Client: `uv` and `uv pip` in `docker/Dockerfile`.
  - Auth: None detected.
  - URLs: `https://download.pytorch.org/whl/rocm7.0`, `https://download.pytorch.org/whl/rocm7.1`, `https://download.pytorch.org/whl/rocm7.2`, and `https://download.pytorch.org/whl/`.

**Container Registries:**
- Docker Hub ROCm image registry - Provides ROCm development base images in `docker/Dockerfile` and `docker/rocm-targets.json`.
  - SDK/Client: Docker CLI via `scripts/run_docker.sh`.
  - Auth: Standard Docker registry auth if required by the local Docker setup; no repository-specific secret is read.
  - Images: `rocm/dev-ubuntu-24.04:7.0.2-complete`, `rocm/dev-ubuntu-24.04:7.1.1-complete`, and `rocm/dev-ubuntu-24.04:7.2-complete`.
- GitHub Container Registry - Provides the `uv` binary image in `docker/Dockerfile`.
  - SDK/Client: Docker build `COPY --from=ghcr.io/astral-sh/uv:0.5.11`.
  - Auth: None detected.

**ROCm Toolchain Commands:**
- ROCm runtime/device discovery - Probed by `src/sol_execbench/core/environment.py`, `src/sol_execbench/driver/problem_packager.py`, and `src/sol_execbench/core/diagnostics.py`.
  - SDK/Client: `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, and `rocm-smi`.
  - Auth: Local system access; `rocm-smi` clock operations may require passwordless `sudo`.
- ROCm profiling - Optional diagnostic profiling in `src/sol_execbench/core/bench/rocm_profiler.py` and routed by `src/sol_execbench/core/toolchain.py`.
  - SDK/Client: `rocprofv3` and `rocprofv3-avail`.
  - Auth: Local ROCm device access through `/dev/kfd` and `/dev/dri`; no network auth.
- Static binary evidence - Optional diagnostic artifact extraction in `src/sol_execbench/core/bench/static_kernel_evidence.py` and `src/sol_execbench/core/toolchain.py`.
  - SDK/Client: `llvm-objdump`, `readelf`, candidate `roc-objdump`, and planned `rga`.
  - Auth: None.

**GPU Runtime Libraries:**
- ROCm HIP runtime and compiler - Required for native HIP/C++ categories and Docker image validation in `docker/Dockerfile`.
  - SDK/Client: `hipcc`, HIP headers/libraries, PyTorch C++ extensions.
  - Auth: Local toolchain only.
- ROCm math/library categories - Supported solution schema categories in `src/sol_execbench/core/data/solution.py`.
  - SDK/Client: hipBLAS, MIOpen, Composable Kernel, and rocWMMA as local ROCm libraries or headers.
  - Auth: None.

## Data Storage

**Databases:**
- Not detected.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only for benchmark inputs, outputs, traces, staging directories, downloaded datasets, profiler artifacts, static evidence sidecars, and compatibility matrices.
- Canonical downloaded benchmark root: `data/SOL-ExecBench/benchmark/` from `scripts/download_solexecbench.py`.
- FlashInfer trace root: `data/flashinfer-trace/` from `scripts/download_data.sh` and Docker runtime wiring in `scripts/run_docker.sh`.
- Temporary evaluation staging: created by `src/sol_execbench/cli/main.py` and populated by `src/sol_execbench/driver/problem_packager.py`.
- Trace and sidecar output: paths selected by CLI flags in `src/sol_execbench/cli/main.py`; sidecars include environment snapshots, `rocprofv3` profiles, and static evidence JSON.

**Caching:**
- Local `uv` cache in Docker: `UV_CACHE_DIR=/home/${HOST_USER}/.cache/uv` in `docker/Dockerfile`.
- Local build cache for tests and builders: `SOLEXECBENCH_CACHE_PATH` is set by `tests/conftest.py`.
- PyTorch/Triton/runtime caches may be used by their libraries, but no external cache service is configured.
- Redis or memcached are not detected.

## Authentication & Identity

**Auth Provider:**
- None for application users.
  - Implementation: CLI executes local benchmark workloads; no login, session, API key, OAuth, SSO, or RBAC layer is implemented in `src/sol_execbench/`.
- External identity is delegated to users' local tools where applicable.
  - Hugging Face: `hf download` in `scripts/download_data.sh` uses the user's local Hugging Face CLI configuration if private access is needed.
  - Docker registries: Docker CLI uses the user's local Docker credential store if registry auth is needed.
  - GitHub Actions: `.github/workflows/code-quality.yml` uses standard GitHub-hosted workflow identity; no custom secrets are referenced.

## Monitoring & Observability

**Error Tracking:**
- None.

**Logs:**
- CLI terminal output through Click/Rich in `src/sol_execbench/cli/main.py`.
- Evaluation subprocess stdout/stderr captured by `subprocess.run` in `src/sol_execbench/cli/main.py`, `scripts/run_dataset.py`, and tests.
- Benchmark trace output as JSONL `Trace` models from `src/sol_execbench/core/data/trace.py`.
- Optional environment snapshot sidecars via `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` in `src/sol_execbench/cli/main.py` and `src/sol_execbench/core/environment.py`.
- Optional `rocprofv3` profile metadata and artifacts through `src/sol_execbench/core/bench/rocm_profiler.py`.
- Optional static kernel evidence sidecars through `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- Docker runtime and dependency evidence through `src/sol_execbench/core/runtime_evidence.py`, `src/sol_execbench/core/dependency_matrix.py`, and `scripts/run_docker.sh`.

## CI/CD & Deployment

**Hosting:**
- Not hosted as a long-running service.
- Runtime distribution is a Python package with console scripts from `pyproject.toml`.
- GPU execution target is local host or Docker container from `docker/Dockerfile`.

**CI Pipeline:**
- GitHub Actions workflow `.github/workflows/code-quality.yml`.
- Matrix: Python `3.12` and `3.13` on `ubuntu-latest`.
- Steps: checkout, setup Python, setup `uv`, `uv sync --locked --all-groups`, `uv run ruff check .`, `uv run ty check`, and CPU-safe pytest subsets.
- GPU/Docker validation is documented in `docs/TESTING.md` and supported by `tests/docker/` and `scripts/run_docker.sh`, but no GPU self-hosted workflow is detected in `.github/workflows/`.

## Environment Configuration

**Required env vars:**
- None required for normal CLI startup.
- ROCm GPU execution requires system-level access to `/dev/kfd` and `/dev/dri`; these are device nodes, not environment variables.
- Native GPU paths require ROCm tools and libraries on PATH / standard library locations, especially `hipcc`, `rocminfo`, and `rocm_agent_enumerator`.

**Important optional env vars:**
- `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` - Environment evidence sidecars in `src/sol_execbench/cli/main.py`.
- `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, `CUDA_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL` - Device visibility and compatibility variables recorded by runtime evidence in `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/runtime_evidence.py`, and `docs/CONFIGURATION.md`.
- `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, and `SOL_EXECBENCH_MCLK_LEVEL` - Clock-lock integration in `src/sol_execbench/core/bench/clock_lock.py` and `docker/entrypoint.sh`.
- `PYTORCH_ALLOC_CONF` - Set to `expandable_segments:True` by `src/sol_execbench/cli/main.py` for compile/evaluation subprocesses.
- `FLASHINFER_TRACE_DIR` - Optional FlashInfer trace lookup root used by `src/sol_execbench/driver/templates/eval_driver.py` and Docker wrapper defaults.
- `IMAGE_NAME`, `IMAGE_TAG`, `ROCM_DOCKER_IMAGE`, and `ROCM_DOCKER_TAG` - Docker image selection in `scripts/run_docker.sh`.
- `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES`, `SOL_EXECBENCH_ALLOW_UNTESTED_TARGET_SMOKE`, `SOL_EXECBENCH_RECORD_CONTAINER_VALIDATION`, `SOL_EXECBENCH_HOST_PYTHON`, `SOL_EXECBENCH_COMPATIBILITY_ENTRY`, `SOL_EXECBENCH_COMPATIBILITY_MATRIX`, and `SOL_EXECBENCH_RUN_DOCKER_DRY_RUN` - Docker wrapper behavior controls in `scripts/run_docker.sh`.
- `SOL_EXECBENCH_DEPENDENCY_*`, `SOL_EXECBENCH_RUNTIME_*`, `SOL_EXECBENCH_DEV_*`, and `SOL_EXECBENCH_DOCKER_*` - Test/debug evidence overrides consumed by `scripts/run_docker.sh`, `src/sol_execbench/core/dependency_matrix.py`, and `src/sol_execbench/core/runtime_evidence.py`.
- Docker build args represented as env-like inputs in `docker/Dockerfile`: `HOST_UID`, `HOST_GID`, `HOST_USER`, `PYTORCH_TORCH_VERSION`, `PYTORCH_TORCHVISION_VERSION`, `PYTORCH_ROCM_INDEX_URL`, `TRITON_ROCM_VERSION`, and `TRITON_ROCM_INDEX_URL`.

**Secrets location:**
- No repository secrets files detected.
- No `.env` files detected in the repository scan.
- Do not store Hugging Face tokens, Docker credentials, proprietary kernels, downloaded datasets, or benchmark outputs in the repository. Use external CLI credential stores and keep downloaded assets under ignored/local data paths.

## Webhooks & Callbacks

**Incoming:**
- None. The project has no HTTP server, webhook receiver, API route, or callback endpoint.

**Outgoing:**
- Hugging Face dataset downloads through `datasets.load_dataset` in `scripts/download_solexecbench.py`.
- Hugging Face Hub CLI downloads through `hf download` in `scripts/download_data.sh`.
- Package resolution/downloads from PyPI and PyTorch wheel indexes through `uv` in `pyproject.toml` and `docker/Dockerfile`.
- Docker image pulls/build stages from Docker Hub and GitHub Container Registry through `docker/Dockerfile` and `scripts/run_docker.sh`.
- No outgoing webhooks, SaaS API writes, telemetry upload, or remote metrics integration detected.

---

*Integration audit: 2026-05-31*
