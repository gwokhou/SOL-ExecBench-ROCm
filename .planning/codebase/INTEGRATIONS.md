# External Integrations

**Analysis Date:** 2026-05-22

## APIs & External Services

**Benchmark Datasets:**
- Hugging Face dataset `nvidia/SOL-ExecBench` - Upstream benchmark data downloaded by `scripts/download_solexecbench.py`.
  - SDK/Client: `datasets.load_dataset` from the `datasets` package.
  - Auth: Not required in code; private or rate-limited Hugging Face access would use external Hugging Face CLI/cache configuration, not repository secrets.
- Hugging Face dataset `flashinfer-ai/flashinfer-trace` - Downloaded by `scripts/download_data.sh` into `data/flashinfer-trace`.
  - SDK/Client: Hugging Face CLI command `hf download`.
  - Auth: Not stored in this repo; use normal Hugging Face CLI credentials outside the codebase when needed.

**Package Registries:**
- PyTorch ROCm wheel index `https://download.pytorch.org/whl/rocm7.1` - Resolves ROCm builds of `torch` and `torchvision` via `[tool.uv.index]` and `[tool.uv.sources]` in `pyproject.toml`.
  - SDK/Client: `uv`.
  - Auth: None.
- PyTorch wheel root `https://download.pytorch.org/whl/` - Resolves `triton-rocm` via `[tool.uv.index]` in `pyproject.toml`.
  - SDK/Client: `uv`.
  - Auth: None.
- PyPI `https://pypi.org/simple` - Resolves non-ROCm Python dependencies in `uv.lock`.
  - SDK/Client: `uv`.
  - Auth: None.

**Container Registries:**
- Docker Hub image `rocm/dev-ubuntu-24.04:7.1.1-complete` - Base image for the ROCm evaluation container in `docker/Dockerfile`.
  - SDK/Client: Docker.
  - Auth: None in code.
- GitHub Container Registry image `ghcr.io/astral-sh/uv:0.5.11` - Source image for copying the `uv` binary in `docker/Dockerfile`.
  - SDK/Client: Docker.
  - Auth: None in code.

**ROCm Host Tooling:**
- ROCm profiler `rocprofv3` - Collects kernel and HIP runtime CSV timing evidence through `src/sol_execbench/core/bench/rocm_profiler.py` and dataset scoring flows in `scripts/run_dataset.py`.
  - SDK/Client: Local subprocess command `rocprofv3`.
  - Auth: None.
- ROCm SMI `rocm-smi` - Locks, verifies, and resets GPU clocks in `src/sol_execbench/core/bench/clock_lock.py`; Docker grants passwordless sudo for SMI commands in `docker/Dockerfile`.
  - SDK/Client: Local subprocess command `rocm-smi`.
  - Auth: Passwordless local `sudo` inside the container for clock control.
- AMD SMI `amd-smi` - Accepted as an SMI tool during container build/runtime readiness checks in `docker/Dockerfile`, `scripts/run_docker.sh`, and `tests/docker/dependencies/test_rocm_runtime.py`.
  - SDK/Client: Local subprocess command `amd-smi`.
  - Auth: Passwordless local `sudo` may be configured in the container.
- ROCm discovery tools `rocm_agent_enumerator` and `rocminfo` - Detect local AMD gfx targets in `src/sol_execbench/driver/problem_packager.py` and `src/sol_execbench/core/diagnostics.py`.
  - SDK/Client: Local subprocess commands.
  - Auth: None.
- HIP compiler `hipcc` - Required for HIP/C++ compilation readiness in `docker/Dockerfile`, `docs/rocm.md`, and `tests/docker/dependencies/test_hip.py`.
  - SDK/Client: Local compiler executable; PyTorch extensions are built through `torch.utils.cpp_extension` in `src/sol_execbench/driver/templates/build_ext.py`.
  - Auth: None.

## Data Storage

**Databases:**
- Not detected.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only for benchmark inputs, downloaded datasets, staging directories, build artifacts, and trace outputs.
- `scripts/download_solexecbench.py` writes SOL ExecBench problems under `data/benchmark/`.
- `scripts/download_data.sh` writes FlashInfer traces under `data/flashinfer-trace/`.
- `src/sol_execbench/driver/problem_packager.py` writes staged `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, source files, `build_ext.py`, `eval_driver.py`, and `benchmark_kernel.so` into a temporary output directory.
- `src/sol_execbench/cli/main.py` writes trace JSONL to the user-specified `--output` path.
- `scripts/run_dataset.py` writes batch outputs under `out/run_dataset/` by default according to `README.md`.

**Caching:**
- `uv` cache is local; Docker sets `UV_CACHE_DIR=/home/${HOST_USER}/.cache/uv` and uses BuildKit cache mounts in `docker/Dockerfile`.
- PyTorch extension builds are staged locally by `src/sol_execbench/driver/templates/build_ext.py`.
- Hugging Face CLI and `datasets` use their standard external caches; no repository cache config is committed.

## Authentication & Identity

**Auth Provider:**
- None. This repository is a local benchmark CLI and does not implement user authentication.
  - Implementation: Not applicable.

**External Credential Handling:**
- Hugging Face dataset access is invoked by `datasets.load_dataset` in `scripts/download_solexecbench.py` and `hf download` in `scripts/download_data.sh`; tokens, if needed, are expected to come from the user's external Hugging Face CLI/cache setup.
- Docker and package registry pulls use public images/indexes by default; no `.env`, `.npmrc`, `.pypirc`, credential, or secret files were detected in the scanned repo root/subtrees.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, OpenTelemetry, Datadog, Prometheus, or similar external error tracking integration detected.

**Logs:**
- CLI output uses Rich tables/progress in `src/sol_execbench/cli/main.py`.
- Evaluation subprocess stdout/stderr is captured by `src/sol_execbench/cli/main.py`; workload logs are stored in trace `Evaluation.log` fields defined by `src/sol_execbench/core/data/trace.py`.
- `src/sol_execbench/core/utils.py` redirects driver stdout/stderr to local files when needed.
- `src/sol_execbench/core/bench/rocm_profiler.py` records profiler command, stdout, stderr, return code, CSV path, parsed rows, and fallback metadata in local evidence payloads.
- `src/sol_execbench/core/diagnostics.py` defines local diagnostic stages and suggested ROCm readiness commands.

## CI/CD & Deployment

**Hosting:**
- Not applicable; no web application or hosted service deployment is detected.
- Runtime delivery is local Python CLI execution and optional Docker-based ROCm evaluation through `docker/Dockerfile` and `scripts/run_docker.sh`.

**CI Pipeline:**
- Not detected in the scanned files. No `.github/workflows/` content appeared in the technology scan.
- Test and readiness commands are documented in `README.md`, `docs/TESTING.md`, and `tests/docker/dependencies/`.

## Environment Configuration

**Required env vars:**
- No required application-level environment variables are detected; `docs/CONFIGURATION.md` states package behavior is configured through CLI flags, optional `config.json`, package metadata, and Docker variables.
- Container runtime variables from `docker/Dockerfile`: `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, `LD_LIBRARY_PATH`, `UV_CACHE_DIR`, `UV_LINK_MODE`, `UV_COMPILE_BYTECODE`, `UV_PYTHON_DOWNLOADS`, `UV_PROJECT_ENVIRONMENT`, `PATH`, and `PYTHONPATH`.
- Docker launcher variables from `scripts/run_docker.sh`: `IMAGE_NAME`, `IMAGE_TAG`, `FLASHINFER_TRACE_DIR`, `SOL_EXECBENCH_GPU_CLK_MHZ`, and `SOL_EXECBENCH_DRAM_CLK_MHZ`.
- Clock-control variables from `src/sol_execbench/core/bench/clock_lock.py` and `docker/entrypoint.sh`: `SOL_EXECBENCH_CLOCKS_LOCKED`, optional `SOL_EXECBENCH_SCLK_LEVEL`, and optional `SOL_EXECBENCH_MCLK_LEVEL`.
- Subprocess memory setting from `src/sol_execbench/cli/main.py`: `PYTORCH_ALLOC_CONF=expandable_segments:True`.

**Secrets location:**
- Not applicable. No repo-local `.env*`, credential, or secret files were detected during the scan, and no source file reads secrets directly.
- External Hugging Face or registry credentials, if used, should remain in user-level tooling outside the repository.

## Webhooks & Callbacks

**Incoming:**
- None. No HTTP server, webhook route, or callback endpoint is implemented.

**Outgoing:**
- Dataset downloads to Hugging Face through `scripts/download_solexecbench.py` and `scripts/download_data.sh`.
- Package/index access to PyPI and PyTorch wheel indexes through `uv` using `pyproject.toml` and `uv.lock`.
- Container image pulls from Docker Hub and GitHub Container Registry through `docker/Dockerfile`.
- No outgoing application webhooks are detected.

---

*Integration audit: 2026-05-22*
