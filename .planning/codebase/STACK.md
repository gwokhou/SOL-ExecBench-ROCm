# Technology Stack

**Analysis Date:** 2026-05-22

## Languages

**Primary:**
- Python 3.12+ - Package source, CLI, schemas, benchmark drivers, dataset scripts, and tests live under `src/sol_execbench/`, `scripts/`, and `tests/`; `pyproject.toml` requires `>=3.12,<3.14` and `.python-version` pins local development to `3.12`.

**Secondary:**
- Bash - Docker launcher and entrypoint scripts in `scripts/run_docker.sh`, `scripts/download_data.sh`, and `docker/entrypoint.sh`.
- HIP/C++ - User-submitted native solution sources are compiled through the staging template in `src/sol_execbench/driver/templates/build_ext.py`; examples live under `examples/hip_cpp/` and tests under `tests/sol_execbench/samples/`.
- Triton Python - Triton ROCm examples live under `examples/triton/`, with dependency readiness covered by `tests/docker/dependencies/test_triton_rocm.py`.
- JSON/JSONL - Benchmark definitions, workloads, solutions, traces, and configs are consumed by `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/data/`, and `src/sol_execbench/driver/problem_packager.py`.

## Runtime

**Environment:**
- Python `>=3.12,<3.14` from `pyproject.toml`; `.python-version` records `3.12`.
- ROCm runtime baseline is ROCm 7.0+ in docs, with the Docker image using `rocm/dev-ubuntu-24.04:7.1.1-complete` in `docker/Dockerfile`.
- PyTorch ROCm exposes AMD GPU devices through the historical `torch.cuda` API; runtime checks appear in `README.md`, `src/sol_execbench/core/utils.py`, and `src/sol_execbench/driver/templates/eval_driver.py`.
- Host/container ROCm tools required for full GPU evaluation include `hipcc`, `rocminfo`, `rocprofv3`, and `rocm-smi` or `amd-smi`, checked in `docker/Dockerfile`, `tests/docker/dependencies/test_rocm_runtime.py`, and `docs/rocm.md`.

**Package Manager:**
- `uv` - Project install and lock management via `uv.lock` and `pyproject.toml`; Docker copies `uv` `0.5.11` from `ghcr.io/astral-sh/uv:0.5.11` in `docker/Dockerfile`.
- Lockfile: present as `uv.lock`.
- Build backend: Hatchling via `[build-system]` in `pyproject.toml`.

## Frameworks

**Core:**
- PyTorch `2.10.0+rocm7.1` on Linux/Windows, `2.10.0` elsewhere - Tensor runtime, reference execution, HIP-backed device events, and native extension builds; declared in `pyproject.toml` and locked in `uv.lock`.
- torchvision `0.25.0+rocm7.1` on Linux/Windows, `0.25.0` elsewhere - PyTorch companion package declared in `pyproject.toml`.
- Pydantic `2.12.5` - Strongly typed benchmark schemas in `src/sol_execbench/core/data/`.
- Click `8.3.1` - CLI commands in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`.
- Rich `14.3.3` - CLI tables/progress output in `src/sol_execbench/cli/main.py`.
- Triton ROCm `3.6.0` - Triton solution category and examples under `examples/triton/`.
- safetensors `0.7.0` - Tensor input loading support referenced by package dependencies and benchmark data docs.
- Hugging Face datasets `4.8.2` - SOL ExecBench dataset download script in `scripts/download_solexecbench.py`.

**Testing:**
- pytest `9.0.2` - Test runner for `tests/`.
- pytest-xdist `3.8.0` - Parallel test execution; `pyproject.toml` sets `addopts = "-n auto --dist loadgroup"`.
- Ruff - Lint/format tool configured in `pyproject.toml`; command guidance is in `AGENTS.md`, `README.md`, and `docs/DEVELOPMENT.md`.

**Build/Dev:**
- Hatchling - Python package build backend configured in `pyproject.toml`.
- Ninja `1.13.0` - Native extension build helper declared in `pyproject.toml`.
- `torch.utils.cpp_extension` - HIP/C++ build path in `src/sol_execbench/driver/templates/build_ext.py`.
- Docker - ROCm GPU evaluation environment in `docker/Dockerfile` and `scripts/run_docker.sh`.
- ROCm tools - `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `rocm-smi`, `amd-smi`, and `rocprofv3` are called from `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and `src/sol_execbench/core/diagnostics.py`.

## Key Dependencies

**Critical:**
- `torch` `2.10.0+rocm7.1` - Required for tensor execution, HIP-backed timing, reference evaluation, and extension builds.
- `triton-rocm` `3.6.0` - Required for Triton ROCm examples and solution execution on Linux.
- `pydantic` `2.12.5` - Enforces public schemas for definitions, workloads, solutions, traces, and configuration.
- `click` `8.3.1` and `rich` `14.3.3` - CLI interface and human-readable benchmark reporting.
- `datasets` `4.8.2` - Downloads the upstream `nvidia/SOL-ExecBench` benchmark dataset in `scripts/download_solexecbench.py`.
- `safetensors` `0.7.0` - Supports safetensors-backed benchmark inputs in `src/sol_execbench/core/bench/io.py`.
- `ninja` `1.13.0` - Speeds and supports PyTorch native extension compilation.

**Infrastructure:**
- `torch-c-dlpack-ext` `0.1.5` - Runtime dependency declared for DLPack interop support in `pyproject.toml`.
- `apache-tvm-ffi` `0.1.9` - Runtime dependency declared for TVM FFI support in `pyproject.toml`.
- `torchvision` `0.25.0+rocm7.1` - Companion package resolved from the ROCm wheel index.
- `pytest` and `pytest-xdist` - Development dependency group in `pyproject.toml`.
- ROCm base image `rocm/dev-ubuntu-24.04:7.1.1-complete` - Containerized runtime in `docker/Dockerfile`.
- `ghcr.io/astral-sh/uv:0.5.11` - Docker build source for the `uv` binary in `docker/Dockerfile`.

## Configuration

**Environment:**
- No `.env`, `.env.example`, or `.env.sample` files are present in the repository scan; `docs/CONFIGURATION.md` states there are no required package-level environment variables.
- Docker runtime variables are defined in `docker/Dockerfile`: `ROCM_PATH=/opt/rocm`, `HIP_PATH=/opt/rocm`, `HIP_PLATFORM=amd`, `LD_LIBRARY_PATH=/opt/rocm/lib`, `UV_CACHE_DIR`, `UV_LINK_MODE`, `UV_COMPILE_BYTECODE`, `UV_PYTHON_DOWNLOADS`, `UV_PROJECT_ENVIRONMENT`, `PATH`, and `PYTHONPATH`.
- `scripts/run_docker.sh` passes `FLASHINFER_TRACE_DIR`, `SOL_EXECBENCH_GPU_CLK_MHZ`, and `SOL_EXECBENCH_DRAM_CLK_MHZ` into the container.
- `docker/entrypoint.sh` sets `SOL_EXECBENCH_CLOCKS_LOCKED` after trying to lock GPU clocks.
- `src/sol_execbench/core/bench/clock_lock.py` reads optional `SOL_EXECBENCH_SCLK_LEVEL` and `SOL_EXECBENCH_MCLK_LEVEL` overrides.
- `src/sol_execbench/cli/main.py` sets `PYTORCH_ALLOC_CONF=expandable_segments:True` for compile and evaluation subprocesses.

**Build:**
- `pyproject.toml` configures package metadata, console scripts, dependencies, dependency groups, pytest options, Ruff exclusions, and ROCm wheel indexes.
- `uv.lock` pins resolved dependency versions.
- `docker/Dockerfile` defines the ROCm evaluation image and installs dependencies with `uv sync --frozen --all-groups`.
- `scripts/run_docker.sh` builds and runs the container with `/dev/kfd`, `/dev/dri`, `--group-add video`, `seccomp=unconfined`, `--ipc=host`, and project volume mounts.
- `src/sol_execbench/driver/templates/build_ext.py` compiles HIP/C++ sources into `benchmark_kernel.so`.
- `src/sol_execbench/core/bench/config/benchmark_config.py` provides benchmark defaults loaded from optional `config.json` files.

## Platform Requirements

**Development:**
- Python 3.12+ and `uv`.
- ROCm 7.0+ host for real GPU evaluation; PyTorch ROCm wheels are selected through `pyproject.toml`.
- AMD GPU access through `/dev/kfd` and `/dev/dri` for Docker runs; `scripts/run_docker.sh` rejects Docker Desktop contexts because ROCm device passthrough requires a native Linux daemon.
- HIP compiler and ROCm tools (`hipcc`, `rocminfo`, `rocprofv3`, `rocm-smi` or `amd-smi`) for native extension builds, diagnostics, profiling, and clock management.

**Production:**
- Not a deployed web/service production system; production-like use is local or containerized benchmark execution.
- Docker target is `rocm/dev-ubuntu-24.04:7.1.1-complete` with mounted source and data directories.
- Supported benchmark hardware targets in schemas are `gfx1200`, `gfx940`, `gfx941`, `gfx942`, and `LOCAL` in `src/sol_execbench/core/data/solution.py`; README states current local validation evidence covers RDNA 4 `gfx1200`, with CDNA 3 validation deferred.

---

*Stack analysis: 2026-05-22*
