# External Integrations

**Analysis Date:** 2026-05-22

## APIs & External Services

**Package Indexes:**
- PyTorch ROCm wheel index - Supplies ROCm-specific `torch`, `torchvision`, and `triton-rocm` wheels.
  - SDK/Client: uv package resolver configured in `pyproject.toml`.
  - Auth: None detected.
  - URLs: `https://download.pytorch.org/whl/rocm7.1` and `https://download.pytorch.org/whl/`.
- PyPI/default uv indexes - Supplies general Python dependencies such as `pydantic`, `click`, `rich`, `datasets`, `safetensors`, `pytest`, and `ninja`.
  - SDK/Client: uv.
  - Auth: None detected.

**Dataset Sources:**
- Hugging Face dataset `nvidia/SOL-ExecBench` - Source benchmark data downloaded and converted by `scripts/download_solexecbench.py`.
  - SDK/Client: `datasets.load_dataset` from the `datasets` package.
  - Auth: Optional Hugging Face local auth may be used by the external library/CLI, but no repository env var or token file is read by project code.
  - Subsets: `L1`, `L2`, `Quant`, and `FlashInfer-Bench` in `scripts/download_solexecbench.py`.
- Hugging Face dataset `flashinfer-ai/flashinfer-trace` - Safetensor/blob dataset downloaded by `scripts/download_data.sh` and mounted/read as `data/flashinfer-trace`.
  - SDK/Client: Hugging Face CLI command `hf download`.
  - Auth: Optional external Hugging Face CLI auth; no project-managed secret.
  - Revision: `1.0` in `scripts/download_data.sh`.

**Container Registries:**
- Docker Hub / ROCm image registry - Base image `rocm/dev-ubuntu-24.04:7.1.1-complete` in `docker/Dockerfile`.
  - SDK/Client: Docker.
  - Auth: None detected in repository.
- GitHub Container Registry - uv binary image `ghcr.io/astral-sh/uv:0.5.11` copied in `docker/Dockerfile`.
  - SDK/Client: Docker.
  - Auth: None detected in repository.

**Hardware / System Tools:**
- ROCm runtime tools - `rocminfo`, `rocm_agent_enumerator`, `hipcc`, `rocprofv3`, `rocm-smi`, and `amd-smi` integrate with local ROCm installations.
  - SDK/Client: subprocess calls in `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/core/diagnostics.py`, `docker/entrypoint.sh`, and `scripts/run_docker.sh`.
  - Auth: `rocm-smi` clock changes require passwordless sudo inside the container, configured in `docker/Dockerfile`.

## Data Storage

**Databases:**
- Not detected.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only.
  - Benchmark source data is expected under `data/` per `README.md`.
  - SOL ExecBench converted problems are written to `data/benchmark/<subset>/<problem>/` by `scripts/download_solexecbench.py`.
  - FlashInfer traces are downloaded to `data/flashinfer-trace` by `scripts/download_data.sh`.
  - CLI staging directories are created with `tempfile.mkdtemp(prefix="sol_execbench_")` in `src/sol_execbench/cli/main.py`.
  - Dataset run outputs default to `out/` in `scripts/run_dataset.py`.
  - Trace outputs are written to user-specified JSONL paths by `src/sol_execbench/cli/main.py`.

**Caching:**
- uv cache at `/home/${HOST_USER}/.cache/uv` in `docker/Dockerfile`.
- Docker build cache mounts for apt and uv in `docker/Dockerfile`.
- PyTorch CUDA/HIP allocator behavior is configured through `PYTORCH_ALLOC_CONF=expandable_segments:True` in `src/sol_execbench/cli/main.py`.
- PyTorch GPU memory cache is explicitly cleared between workloads via `torch.cuda.empty_cache()` in `src/sol_execbench/driver/templates/eval_driver.py`.
- No Redis, Memcached, or network cache service detected.

## Authentication & Identity

**Auth Provider:**
- None.
  - Implementation: No user login, sessions, OAuth, API keys, or identity provider integration detected in `src/`, `scripts/`, `docker/`, or `pyproject.toml`.

**Local Privilege Boundary:**
- Docker image grants passwordless sudo for `amd-smi` or `rocm-smi` only, when present, in `docker/Dockerfile`.
- Clock locking calls `sudo -n rocm-smi` in `src/sol_execbench/core/bench/clock_lock.py`.
- Docker wrapper validates host ROCm devices and native Docker context before container launch in `scripts/run_docker.sh`.

## Monitoring & Observability

**Error Tracking:**
- None.

**Logs:**
- CLI status, tables, and progress use Rich console output in `src/sol_execbench/cli/main.py`.
- Evaluation driver redirects library/JIT noise to stderr and emits strict JSON trace objects to stdout in `src/sol_execbench/driver/templates/eval_driver.py`.
- Failed dataset CLI invocations write local log files named `<job_name>_cli.log` under the run output directory in `scripts/run_dataset.py`.
- Stage-aware diagnostic messages are modeled by `StageDiagnostic` and `SolExecBenchError` in `src/sol_execbench/core/diagnostics.py`.
- Benchmark trace objects capture evaluation status, correctness, performance, device, and library/tool versions through models in `src/sol_execbench/core/data/trace.py` and helpers in `src/sol_execbench/core/bench/utils.py`.
- Profiling readiness is descriptive only and selects among `rocprofv3`, `rocprofiler-compute`, `omniperf`, or skip in `src/sol_execbench/core/diagnostics.py`.

## CI/CD & Deployment

**Hosting:**
- Not applicable for the application runtime; this is a local/container benchmark package.
- Docker runtime is defined by `docker/Dockerfile` and launched by `scripts/run_docker.sh`.

**CI Pipeline:**
- No `.github/workflows/` files detected in the repository.
- Test and validation commands are documented in `README.md`, `AGENTS.md`, and `docs/rocm.md`.
- Docker dependency smoke tests live in `tests/docker/dependencies/`.

## Environment Configuration

**Required env vars:**
- `FLASHINFER_TRACE_DIR` - Optional/expected path to FlashInfer trace data; forwarded by `scripts/run_docker.sh` and read by `src/sol_execbench/driver/templates/eval_driver.py`.
- `PYTORCH_ALLOC_CONF` - Set to `expandable_segments:True` for compile/eval subprocesses in `src/sol_execbench/cli/main.py`.
- `ROCM_PATH` - Set to `/opt/rocm` in `docker/Dockerfile`.
- `HIP_PATH` - Set to `/opt/rocm` in `docker/Dockerfile`.
- `HIP_PLATFORM` - Set to `amd` in `docker/Dockerfile`.
- `LD_LIBRARY_PATH` - Set to `/opt/rocm/lib` in `docker/Dockerfile`.
- `SOL_EXECBENCH_CLOCKS_LOCKED` - Set by `docker/entrypoint.sh` and read by `src/sol_execbench/core/bench/clock_lock.py` and `src/sol_execbench/driver/templates/eval_driver.py`.
- `SOL_EXECBENCH_SCLK_LEVEL` - Optional SCLK DPM override read by `src/sol_execbench/core/bench/clock_lock.py`.
- `SOL_EXECBENCH_MCLK_LEVEL` - Optional MCLK DPM override read by `src/sol_execbench/core/bench/clock_lock.py`.
- `SOL_EXECBENCH_GPU_CLK_MHZ` and `SOL_EXECBENCH_DRAM_CLK_MHZ` - Forwarded by `scripts/run_docker.sh`; no direct Python reader detected.
- `IMAGE_NAME` and `IMAGE_TAG` - Optional Docker wrapper overrides in `scripts/run_docker.sh`.

**Secrets location:**
- No repository-managed secrets detected.
- `.env*`, credential, secret, package auth, and key files were not found at repo depth checked.
- Hugging Face authentication, if needed, is external to this repo through the user environment or Hugging Face CLI.

## Webhooks & Callbacks

**Incoming:**
- None detected. No HTTP server, webhook route, or callback endpoint exists in `src/`, `scripts/`, or `docker/`.

**Outgoing:**
- Dataset downloads through Hugging Face APIs/CLI in `scripts/download_solexecbench.py` and `scripts/download_data.sh`.
- Dependency downloads through uv package indexes configured in `pyproject.toml`.
- Docker image pulls from `rocm/dev-ubuntu-24.04:7.1.1-complete` and `ghcr.io/astral-sh/uv:0.5.11` in `docker/Dockerfile`.
- No application runtime webhooks or outbound business API calls detected.

---

*Integration audit: 2026-05-22*
