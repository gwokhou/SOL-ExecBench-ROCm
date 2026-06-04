# External Integrations

**Analysis Date:** 2026-06-04

## APIs & External Services

**Dataset Acquisition:**
- Hugging Face Datasets - Downloads the public `nvidia/SOL-ExecBench` dataset into local benchmark layout.
  - SDK/Client: `datasets.load_dataset` from `datasets>=4.8.2`
  - Auth: Hugging Face token is not read directly by repository code; users rely on standard Hugging Face CLI/cache behavior when needed.
  - Implementation: `scripts/download_solexecbench.py`
  - Local output: `data/SOL-ExecBench/benchmark/`

**Package Indexes:**
- PyPI - Default package index for uv dependency resolution.
  - SDK/Client: uv configured by `pyproject.toml`
  - Auth: Not detected
- PyTorch ROCm wheel indexes - ROCm-specific PyTorch, torchvision, and Triton ROCm wheels.
  - SDK/Client: uv indexes in `pyproject.toml`; Docker build args in `docker/Dockerfile`
  - Auth: Not detected
  - URLs: `https://download.pytorch.org/whl/rocm7.1` and `https://download.pytorch.org/whl/`

**Container Registry:**
- Docker Hub ROCm base images - Container user-space source for benchmark runs.
  - SDK/Client: Docker CLI invoked by `scripts/run_docker.sh`
  - Auth: Not detected
  - Images: `rocm/dev-ubuntu-24.04` tags declared in `docker/rocm-targets.json`
- GitHub Container Registry - uv binary source in Docker builds.
  - SDK/Client: Docker multi-stage copy from `ghcr.io/astral-sh/uv:0.11.18`
  - Auth: Not detected
  - Implementation: `docker/Dockerfile`

## Data Storage

**Databases:**
- Not detected
  - Connection: Not applicable
  - Client: Not applicable

**File Storage:**
- Local filesystem only.
  - Benchmark assets: `data/`
  - Downloaded SOL-ExecBench layout: `data/SOL-ExecBench/benchmark/`
  - Trace JSONL output: CLI `--output` path from `src/sol_execbench/cli/main.py`
  - Environment/profile/static-evidence sidecars: generated next to trace output or staging directories by `src/sol_execbench/cli/main.py`
  - Native build staging: temporary or retained directories managed by `src/sol_execbench/driver/problem_packager.py`

**Caching:**
- uv cache - `UV_CACHE_DIR` is configured as `/home/${HOST_USER}/.cache/uv` in `docker/Dockerfile`; local commands often use `UV_CACHE_DIR=/tmp/uv-cache` in `docs/`.
- PyTorch extension build cache - Native solution builds go through `torch.utils.cpp_extension.load` in `src/sol_execbench/driver/templates/build_ext.py`; build artifacts are staged as `benchmark_kernel.so`.
- Hugging Face cache - Used indirectly by `datasets.load_dataset` in `scripts/download_solexecbench.py`; no repo-specific cache path is defined.

## Authentication & Identity

**Auth Provider:**
- None for application runtime.
  - Implementation: The CLI is local and does not implement login, sessions, user identity, or authorization middleware.
- Hugging Face authentication may be required externally for dataset access depending on user environment.
  - Implementation: Standard Hugging Face tooling outside repository code; `scripts/download_data.sh` documents using `huggingface-hub[cli]`, and `scripts/download_solexecbench.py` relies on `datasets.load_dataset`.

## GPU, Profiler & Toolchain Integrations

**ROCm Runtime:**
- HIP/PyTorch ROCm - The benchmark uses PyTorch ROCm devices through historical `torch.cuda` compatibility APIs.
  - Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/environment.py`
  - Device access: `/dev/kfd` and `/dev/dri` are probed by `scripts/run_docker.sh`
  - Device visibility: `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION` are captured by `src/sol_execbench/core/environment.py`

**Native HIP/C++ Build:**
- PyTorch extension build system - Compiles HIP/C++ and ROCm library solution categories into `benchmark_kernel.so`.
  - Client: `torch.utils.cpp_extension.load`
  - Files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`
  - Architecture selection: `--offload-arch=<gfx>` and `PYTORCH_ROCM_ARCH` from solution `target_hardware`
  - Local gfx detection: `rocm_agent_enumerator -name` and `rocminfo` in `src/sol_execbench/driver/problem_packager.py`

**ROCm Libraries and Categories:**
- hipBLAS, MIOpen, Composable Kernel, and rocWMMA are supported solution categories.
  - Schema: `src/sol_execbench/core/data/solution.py`
  - Examples: `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, `examples/rocwmma/`
  - Diagnostics: `src/sol_execbench/core/diagnostics.py`

**Profiler:**
- `rocprofv3` - Optional diagnostic profile and timing evidence collection.
  - Client: subprocess command builder in `src/sol_execbench/core/bench/rocm_profiler.py`
  - CLI flag: `--profile rocprofv3` in `src/sol_execbench/cli/main.py`
  - Evidence schemas: `sol_execbench.rocprofv3_profile.v1` and `sol_execbench.rocprofv3_timing.v1`
  - Artifacts: `.rocprofv3/` output directories and profile sidecar JSON near trace output

**Environment and Clock Tools:**
- `amd-smi`, `rocminfo`, and `rocm_agent_enumerator` - Bounded environment probes in `src/sol_execbench/core/environment.py`.
- `amd-smi` or `rocm-smi` - Clock locking and Docker entrypoint clock status in `docker/entrypoint.sh` and `src/sol_execbench/core/bench/clock_lock.py`.
- `hipcc` - Toolchain/version probe in `docker/Dockerfile` and dependency evidence collection in `src/sol_execbench/core/dependency_matrix.py`.
- `llvm-objdump` and `readelf` - Static kernel evidence extraction routes documented in `docs/rocm_toolchain_routing.md` and collected through `src/sol_execbench/core/bench/static_kernel_evidence.py`.

## Monitoring & Observability

**Error Tracking:**
- None

**Logs:**
- Local CLI stderr/stdout using Click, Rich, and subprocess output capture in `src/sol_execbench/cli/main.py`.
- No-trace diagnostic sidecars with bounded stdout/stderr tails are written by `src/sol_execbench/cli/main.py`.
- Environment diagnostics are produced by `sol-execbench doctor` through `src/sol_execbench/core/environment.py`.
- Compatibility, dependency, runtime, profiler, static-evidence, and scoring reports are emitted as local JSON/Markdown artifacts by modules under `src/sol_execbench/core/` and scripts under `scripts/`.

## CI/CD & Deployment

**Hosting:**
- Not applicable. This repository provides a local Python package, CLI tools, scripts, examples, and Docker workflow.

**CI Pipeline:**
- GitHub Actions is present.
  - Workflow: `.github/workflows/code-quality.yml`
  - Purpose: code quality/test automation for the repository
- Local hooks are configured by `.pre-commit-config.yaml`.

## Environment Configuration

**Required env vars:**
- None for CPU-safe metadata commands and schema tests.
- ROCm GPU evaluation depends on host/device availability rather than application secrets.

**Important runtime env vars:**
- `PYTORCH_ALLOC_CONF` - Set to `expandable_segments:True` by `src/sol_execbench/cli/main.py` for staged compile/eval subprocesses.
- `PYTORCH_ROCM_ARCH` - Set by `src/sol_execbench/driver/templates/build_ext.py` when solution targets concrete `gfx*` hardware and the variable is unset.
- `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` - Enable and route environment snapshot sidecars in `src/sol_execbench/cli/main.py`.
- `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION` - Captured as GPU visibility/runtime evidence in `src/sol_execbench/core/environment.py` and `scripts/run_docker.sh`.
- `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, `SOL_EXECBENCH_MCLK_LEVEL`, `SOL_EXECBENCH_GPU_CLK_MHZ`, `SOL_EXECBENCH_DRAM_CLK_MHZ` - Clock and timing diagnostics in `docker/entrypoint.sh`, `src/sol_execbench/core/bench/clock_lock.py`, and `scripts/run_docker.sh`.
- `IMAGE_NAME`, `IMAGE_TAG`, `ROCM_DOCKER_IMAGE`, `ROCM_DOCKER_TAG` - Docker image naming and unsafe target overrides in `scripts/run_docker.sh`.
- `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES`, `SOL_EXECBENCH_ALLOW_UNTESTED_TARGET_SMOKE`, `SOL_EXECBENCH_RECORD_CONTAINER_VALIDATION` - Docker target and compatibility decision toggles in `scripts/run_docker.sh`.
- `SOL_EXECBENCH_COMPATIBILITY_ENTRY`, `SOL_EXECBENCH_COMPATIBILITY_MATRIX`, `SOL_EXECBENCH_DEPENDENCY_*`, `SOL_EXECBENCH_RUNTIME_*`, `SOL_EXECBENCH_DEV_*`, `SOL_EXECBENCH_DOCKER_*` - Docker/dependency/runtime evidence override and sidecar controls in `scripts/run_docker.sh`.
- `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, `LD_LIBRARY_PATH`, `UV_CACHE_DIR`, `UV_LINK_MODE`, `UV_COMPILE_BYTECODE`, `UV_PYTHON_DOWNLOADS`, `UV_PROJECT_ENVIRONMENT` - Docker environment configured in `docker/Dockerfile`.

**Secrets location:**
- No `.env` or secret files detected during mapping.
- Do not commit Hugging Face tokens, credentials, proprietary kernels, or downloaded datasets; repository policy is documented in `README.md`, `docs/provenance.md`, and `docs/compliance.md`.

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-06-04*
