# External Integrations

**Analysis Date:** 2026-05-24

## APIs & External Services

**Dataset Distribution:**
- Hugging Face Hub dataset `nvidia/SOL-ExecBench` - Source for canonical benchmark definitions, references, and workloads.
  - SDK/Client: `datasets.load_dataset` from `datasets>=4.8.2` in `scripts/download_solexecbench.py`.
  - Auth: Optional Hugging Face CLI/user cache outside the repo; no repository env var or token file is required by source code.
- Hugging Face Hub dataset `flashinfer-ai/flashinfer-trace` - Source for FlashInfer trace safetensors used by workloads.
  - SDK/Client: `hf download` CLI in `scripts/download_data.sh`.
  - Auth: Optional Hugging Face CLI/user cache outside the repo; no repository env var or token file is required by source code.

**Package Indexes:**
- PyPI - Default Python package index.
  - SDK/Client: `uv` configured by `[[tool.uv.index]]` in `pyproject.toml`.
  - Auth: Not detected.
- PyTorch ROCm wheel index `https://download.pytorch.org/whl/rocm7.1` - Linux/Windows source for `torch==2.10.0+rocm7.1` and `torchvision==0.25.0+rocm7.1`.
  - SDK/Client: `uv` source mapping in `pyproject.toml`.
  - Auth: Not detected.
- PyTorch ROCm root wheel index `https://download.pytorch.org/whl/` - Source for `triton-rocm==3.6.0` on Linux.
  - SDK/Client: `uv` source mapping in `pyproject.toml`.
  - Auth: Not detected.

**Container Registries:**
- Docker Hub or configured OCI registry for `rocm/dev-ubuntu-24.04:7.1.1-complete` - Base ROCm development image in `docker/Dockerfile`.
  - SDK/Client: Docker build invoked through `scripts/run_docker.sh`.
  - Auth: Not detected.
- GitHub Container Registry `ghcr.io/astral-sh/uv:0.5.11` - Supplies the `uv` binary during Docker builds in `docker/Dockerfile`.
  - SDK/Client: Docker multi-stage `COPY --from`.
  - Auth: Not detected.

**ROCm Host Tooling:**
- ROCm runtime tools - Hardware discovery, profiling, native compilation, and clock locking.
  - SDK/Client: `rocminfo`, `rocm_agent_enumerator`, `rocm-smi`, `amd-smi`, `hipcc`, and `rocprofv3`.
  - Auth: `sudo -n rocm-smi` is required only for clock locking in `src/sol_execbench/core/bench/clock_lock.py` and `docker/entrypoint.sh`.

## Data Storage

**Databases:**
- Not detected.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only.
  - Benchmark dataset layout: `data/SOL-ExecBench/benchmark`.
  - FlashInfer trace assets: `data/flashinfer-trace`.
  - CLI staging directories: temporary `sol_execbench_*` directories created by `src/sol_execbench/cli/main.py`.
  - Output traces: user-selected `--output` JSONL path in `src/sol_execbench/cli/main.py`.
  - Dataset run output: `out/run_dataset/` default in `scripts/run_dataset.py`.
  - Safetensors lookup roots: staging directory and optional `FLASHINFER_TRACE_DIR` in `src/sol_execbench/driver/templates/eval_driver.py`.

**Caching:**
- `uv` dependency cache in Docker: `UV_CACHE_DIR=/home/${HOST_USER}/.cache/uv` in `docker/Dockerfile`.
- Docker build cache mounts for apt and `uv` in `docker/Dockerfile`.
- PyTorch allocator behavior configured per subprocess with `PYTORCH_ALLOC_CONF=expandable_segments:True` in `src/sol_execbench/cli/main.py`.
- No Redis, Memcached, or external cache service detected.

## Authentication & Identity

**Auth Provider:**
- None for the application.
  - Implementation: Local CLI tool; no user accounts, sessions, OAuth, JWT, or API auth paths are detected in `src/sol_execbench/`.

**Host Privilege Boundary:**
- ROCm clock locking uses passwordless `sudo` for `rocm-smi`.
  - Implementation: `src/sol_execbench/core/bench/clock_lock.py` runs `sudo -n rocm-smi` commands; `docker/Dockerfile` adds a sudoers entry for `amd-smi` or `rocm-smi`; `docker/entrypoint.sh` locks and unlocks clocks around container execution.

## Monitoring & Observability

**Error Tracking:**
- None.

**Logs:**
- CLI status, tables, and runtime logs use Rich/Click output in `src/sol_execbench/cli/main.py`.
- Evaluation traces are emitted as JSONL `Trace` objects by `src/sol_execbench/driver/templates/eval_driver.py`.
- `rocprofv3` CSV outputs are parsed into timing evidence by `src/sol_execbench/core/bench/rocm_profiler.py`.
- Dataset batch logs and artifacts are written by `scripts/run_dataset.py`.
- Python logging is used for ROCm clock-lock warnings in `src/sol_execbench/core/bench/clock_lock.py`.

## CI/CD & Deployment

**Hosting:**
- Not applicable; this is a local/ROCm-container benchmark tool.
- Runtime container is built locally from `docker/Dockerfile` through `scripts/run_docker.sh`.

**CI Pipeline:**
- Not detected in repository files inspected; no `.github/workflows/` files are present in the scanned tree.
- Local quality hooks are defined in `.pre-commit-config.yaml` for Ruff and DCO sign-off checks.

## Environment Configuration

**Required env vars:**
- None for baseline CLI operation.
- Container-defined: `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, `PATH`, `LD_LIBRARY_PATH`, `UV_CACHE_DIR`, `UV_LINK_MODE`, `UV_COMPILE_BYTECODE`, `UV_PYTHON_DOWNLOADS`, `UV_PROJECT_ENVIRONMENT`, and `PYTHONPATH` in `docker/Dockerfile`.
- Optional runtime: `FLASHINFER_TRACE_DIR`, `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, `SOL_EXECBENCH_MCLK_LEVEL`, `SOL_EXECBENCH_GPU_CLK_MHZ`, `SOL_EXECBENCH_DRAM_CLK_MHZ`, `PYTORCH_ALLOC_CONF`, and `PYTORCH_ROCM_ARCH`; documented in `docs/CONFIGURATION.md` and used by `src/sol_execbench/cli/main.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/core/bench/clock_lock.py`, and `scripts/run_docker.sh`.

**Secrets location:**
- No `.env*`, credential, or secret files were detected during mapping.
- Hugging Face credentials, if needed for downloads, are expected to come from the user's external Hugging Face CLI/cache rather than repository files.
- Docker registry credentials, if needed, are external to the repo.

## Webhooks & Callbacks

**Incoming:**
- None detected. The project exposes CLI entry points `sol-execbench` and `sol-execbench-baseline` in `pyproject.toml`, not HTTP endpoints.

**Outgoing:**
- Hugging Face dataset downloads initiated by `scripts/download_solexecbench.py` and `scripts/download_data.sh`.
- Python package resolution from PyPI and PyTorch ROCm indexes via `uv` in `pyproject.toml`.
- OCI image pulls for `rocm/dev-ubuntu-24.04:7.1.1-complete` and `ghcr.io/astral-sh/uv:0.5.11` during Docker builds in `docker/Dockerfile`.
- No application webhooks or callback URLs detected.

---

*Integration audit: 2026-05-24*
