# External Integrations

**Analysis Date:** 2026-06-01

## APIs & External Services

**Dataset Acquisition:**
- Hugging Face Hub dataset `nvidia/SOL-ExecBench` - public benchmark dataset downloaded by `scripts/download_solexecbench.py` into `data/SOL-ExecBench/benchmark`.
  - SDK/Client: `datasets.load_dataset` from the `datasets>=4.8.2` package in `pyproject.toml`.
  - Auth: no repository-specific environment variable; private or gated Hugging Face access would use standard Hugging Face CLI/cache auth outside this codebase.
- Hugging Face Hub dataset `flashinfer-ai/flashinfer-trace` - optional trace assets downloaded by `scripts/download_data.sh` into `data/flashinfer-trace`.
  - SDK/Client: `hf download` CLI from `huggingface-hub[cli]`, invoked by `scripts/download_data.sh` and documented in `README.md`.
  - Auth: standard Hugging Face CLI/cache auth; no project-specific token variable is read.

**Package Indexes:**
- PyPI - default Python package registry configured as the default uv index in `pyproject.toml`.
  - SDK/Client: `uv`.
  - Auth: not detected.
- PyTorch ROCm wheel indexes - `https://download.pytorch.org/whl/rocm7.1` and `https://download.pytorch.org/whl/` configured in `pyproject.toml`; ROCm `7.0`, `7.1`, and `7.2` target-specific URLs are also declared in `docker/rocm-targets.json`.
  - SDK/Client: `uv`, `uv pip`, and Docker build commands in `docker/Dockerfile`.
  - Auth: none.
- GitHub Container Registry image `ghcr.io/astral-sh/uv:0.5.11` - used as a Docker build source for the uv binary in `docker/Dockerfile`.
  - SDK/Client: Docker build.
  - Auth: none for public pull.
- Docker Hub ROCm images `rocm/dev-ubuntu-24.04` - base images used by `docker/Dockerfile` and selected by `docker/rocm-targets.json`.
  - SDK/Client: Docker.
  - Auth: none for public pull.

**ROCm System Tools:**
- ROCm runtime/device discovery - `rocminfo`, `rocm_agent_enumerator`, and `amd-smi` are probed by `src/sol_execbench/core/environment.py`.
  - SDK/Client: local subprocess commands.
  - Auth: not applicable.
- ROCm clock management - `rocm-smi` is invoked through passwordless `sudo -n` by `src/sol_execbench/core/bench/clock_lock.py`; `docker/entrypoint.sh` exports `SOL_EXECBENCH_CLOCKS_LOCKED` after attempts.
  - SDK/Client: local subprocess commands.
  - Auth: passwordless sudo rule configured in `docker/Dockerfile` for available SMI tools.
- ROCm profiler evidence - `rocprofv3` and `rocprofv3-avail` are modeled in `src/sol_execbench/core/toolchain.py`; `rocprofv3` commands are built and run by `src/sol_execbench/core/bench/rocm_profiler.py`.
  - SDK/Client: local subprocess commands.
  - Auth: not applicable.
- Static artifact inspection - `llvm-objdump`, `readelf`, and candidate `roc-objdump`/`rga` are routed by `src/sol_execbench/core/toolchain.py` and used by `src/sol_execbench/core/bench/static_kernel_evidence.py`.
  - SDK/Client: local subprocess commands.
  - Auth: not applicable.

**Container Runtime:**
- Docker daemon - used by `scripts/run_docker.sh` for image build, preflight classification, and GPU container execution.
  - SDK/Client: `docker build`, `docker run`, `docker context show`, and `docker context inspect` in `scripts/run_docker.sh`.
  - Auth: Docker daemon access; no application-level auth provider.

## Data Storage

**Databases:**
- Not detected. The package does not define a relational, document, vector, or cloud database client.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only for benchmark definitions, workloads, solutions, traces, sidecars, reports, staging directories, Docker manifests, and downloaded datasets.
- Canonical dataset layout is under `data/SOL-ExecBench/benchmark` as implemented by `scripts/download_solexecbench.py`.
- Optional FlashInfer traces are under `data/flashinfer-trace` and referenced through `FLASHINFER_TRACE_DIR` in `src/sol_execbench/driver/templates/eval_driver.py`, `scripts/run_docker.sh`, and `docs/CONFIGURATION.md`.
- CLI trace output and sidecars are written to user-supplied paths through `--output`, `SOLEXECBENCH_ENV_SNAPSHOT_PATH`, `--profile rocprofv3`, and `--static-evidence auto` in `src/sol_execbench/cli/main.py`.

**Caching:**
- uv cache path is configured by `UV_CACHE_DIR` in `docker/Dockerfile`.
- Hugging Face and `datasets` use their standard local caches; project-specific cache configuration is not detected.
- PyTorch extension builds occur in temporary or staged directories created by `src/sol_execbench/driver/problem_packager.py`; tests may isolate `SOLEXECBENCH_CACHE_PATH` in `tests/conftest.py`.

## Authentication & Identity

**Auth Provider:**
- None. There is no application login, OAuth, API-key auth, session management, or identity provider in `src/sol_execbench/`.
  - Implementation: local CLI inputs and local subprocess execution.

**External Credential Surfaces:**
- Hugging Face dataset downloads rely on external Hugging Face CLI/cache credentials when needed; no `HF_TOKEN` or project-specific token handling is implemented in `scripts/download_solexecbench.py` or `scripts/download_data.sh`.
- Docker registry pulls rely on the user's Docker configuration if a registry requires credentials; no project-specific Docker auth code exists.
- ROCm clock locking relies on local sudoers permissions for SMI tools configured in `docker/Dockerfile`, not application identity.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, OpenTelemetry, hosted log aggregator, or metrics backend is detected.

**Logs:**
- CLI diagnostics are emitted to stderr through Rich console output in `src/sol_execbench/cli/main.py`.
- Evaluation subprocess stdout/stderr is captured by `subprocess.run` in `src/sol_execbench/cli/main.py` and converted into trace logs through `ProblemPackager.convert_stdout_to_traces` in `src/sol_execbench/driver/problem_packager.py`.
- Dataset batch runs save CLI logs through helpers in `scripts/run_dataset.py`.
- ROCm environment snapshots, profiler sidecars, static kernel evidence sidecars, compatibility matrices, and report JSON/Markdown files are local diagnostic artifacts produced by `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/compatibility.py`, and `scripts/report_*.py`.

## CI/CD & Deployment

**Hosting:**
- Not detected. This repository ships a local Python package and Docker workflow, not a hosted service.

**CI Pipeline:**
- No CI workflow files are present in the explored tree. Test and lint commands are documented in `README.md`, `docs/DEVELOPMENT.md`, and `docs/TESTING.md`.

**Container Build/Run:**
- `docker/Dockerfile` builds the ROCm benchmark environment.
- `scripts/run_docker.sh` selects targets from `docker/rocm-targets.json`, mounts `/dev/kfd` and `/dev/dri`, forwards selected environment variables, and can write compatibility sidecars.

## Environment Configuration

**Required env vars:**
- None for normal CLI startup, as documented in `docs/CONFIGURATION.md`.

**Optional runtime and diagnostic env vars:**
- `PYTORCH_ALLOC_CONF` - set by `src/sol_execbench/cli/main.py` for compile/eval subprocesses.
- `PYTORCH_ROCM_ARCH` - used by `src/sol_execbench/driver/templates/build_ext.py` to control HIP offload architectures.
- `SOLEXECBENCH_ENV_SNAPSHOT`, `SOLEXECBENCH_ENV_SNAPSHOT_PATH` - enable and locate environment sidecars in `src/sol_execbench/cli/main.py`.
- `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, `CUDA_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL` - visibility/evidence variables collected or forwarded by `src/sol_execbench/core/environment.py`, `scripts/run_docker.sh`, and `docs/CONFIGURATION.md`.
- `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, `SOL_EXECBENCH_MCLK_LEVEL` - clock-lock state and overrides in `src/sol_execbench/core/bench/clock_lock.py`.
- `FLASHINFER_TRACE_DIR` - optional trace asset lookup root in `src/sol_execbench/driver/templates/eval_driver.py` and `scripts/run_docker.sh`.
- `SOL_EXECBENCH_*` Docker, dependency, runtime, compatibility, and preflight overrides - documented in `docs/CONFIGURATION.md` and implemented in `scripts/run_docker.sh`, `src/sol_execbench/core/dependency_matrix.py`, and `src/sol_execbench/core/compatibility.py`.
- `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, `UV_CACHE_DIR`, `UV_LINK_MODE`, `UV_COMPILE_BYTECODE`, `UV_PYTHON_DOWNLOADS`, and `UV_PROJECT_ENVIRONMENT` - Docker image environment in `docker/Dockerfile`.

**Secrets location:**
- Not detected. No `.env`, `.env.*`, `credentials.*`, `secrets.*`, key, certificate, or package-auth files were detected during the mapper scan. Do not add credentials to repository files.

## Webhooks & Callbacks

**Incoming:**
- None. No HTTP server, webhook route, callback endpoint, or background listener is implemented.

**Outgoing:**
- Hugging Face dataset downloads through `datasets.load_dataset` in `scripts/download_solexecbench.py`.
- Hugging Face CLI dataset download through `hf download` in `scripts/download_data.sh`.
- Python package and wheel downloads through uv/PyPI/PyTorch indexes configured in `pyproject.toml`, `docker/Dockerfile`, and `docker/rocm-targets.json`.
- Docker image pulls/builds through `docker/Dockerfile` and `scripts/run_docker.sh`.
- All benchmark execution, profiling, toolchain probing, and clock control integrations are local subprocess calls, not network callbacks.

---

*Integration audit: 2026-06-01*
