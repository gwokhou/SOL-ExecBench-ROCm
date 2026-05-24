# Technology Stack

**Analysis Date:** 2026-05-24

## Languages

**Primary:**
- Python `>=3.12,<3.14` - Package, CLI, evaluator driver, dataset tooling, diagnostics, and tests under `src/sol_execbench/`, `scripts/`, and `tests/`.

**Secondary:**
- HIP/C++ - User solution sources compiled through the native extension path in `src/sol_execbench/driver/templates/build_ext.py`; examples live under `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, and `examples/rocwmma/`.
- Shell - Docker and dataset helper scripts in `scripts/run_docker.sh`, `scripts/download_data.sh`, and `docker/entrypoint.sh`.
- Markdown/JSON/JSONL - Public schemas and docs in `docs/`, sample problems in `tests/sol_execbench/samples/`, and benchmark data layouts under `data/`.

## Runtime

**Environment:**
- Python `>=3.12,<3.14`, declared in `pyproject.toml`.
- ROCm `>=7.0` host runtime, documented in `README.md` and `docs/rocm.md`.
- PyTorch ROCm `2.10.0+rocm7.1` on Linux/Windows through the `pytorch-rocm71` uv index in `pyproject.toml`.
- Docker runtime based on `rocm/dev-ubuntu-24.04:7.1.1-complete` in `docker/Dockerfile`.

**Package Manager:**
- `uv` - Dependency resolver and runner; lockfile is `uv.lock`.
- Docker image copies `uv` from `ghcr.io/astral-sh/uv:0.5.11` in `docker/Dockerfile`.
- Lockfile: present at `uv.lock`.

## Frameworks

**Core:**
- Click `>=8.0` / locked `8.3.1` - CLI commands in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`.
- Rich `>=13.0` / locked `14.3.3` - CLI tables and progress display in `src/sol_execbench/cli/main.py`.
- Pydantic `>=2.12.5` / locked `2.12.5` - Public schema models in `src/sol_execbench/core/data/`.
- PyTorch ROCm `2.10.0+rocm7.1` - Reference execution, user solution execution, timing, tensor IO, and native extension compilation in `src/sol_execbench/driver/templates/eval_driver.py` and `src/sol_execbench/driver/templates/build_ext.py`.
- Triton ROCm `3.6.0` - Supported solution category for generated Triton kernels; timing policy routes Triton to `rocprofv3` in `src/sol_execbench/core/bench/timing_policy.py`.
- Hugging Face Datasets `4.8.2` - Public benchmark dataset downloader in `scripts/download_solexecbench.py`.

**Testing:**
- Pytest `>=9.0.2` / locked `9.0.2` - Test runner configured in `pyproject.toml`.
- pytest-xdist `>=3.5` / locked `3.8.0` - Parallel test execution via `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.

**Build/Dev:**
- Hatchling - Python build backend declared in `pyproject.toml`.
- Ruff - Lint and format tool configured in `pyproject.toml`; pre-commit uses `ruff-pre-commit` `v0.9.2` in `.pre-commit-config.yaml`.
- Ninja `>=1.13.0` / locked `1.13.0` - Build acceleration for PyTorch native extensions.
- `torch.utils.cpp_extension` - HIP/C++ extension build API used by `src/sol_execbench/driver/templates/build_ext.py`.
- Docker - ROCm evaluation container defined by `docker/Dockerfile` and launched by `scripts/run_docker.sh`.

## Key Dependencies

**Critical:**
- `torch==2.10.0+rocm7.1` - Core ROCm tensor runtime and device API; code intentionally uses PyTorch's `torch.cuda` namespace for HIP-backed devices in `src/sol_execbench/driver/templates/eval_driver.py`.
- `torchvision==0.25.0+rocm7.1` - Locked alongside PyTorch ROCm in `pyproject.toml` for compatibility.
- `triton-rocm==3.6.0` - Triton ROCm solution support on Linux.
- `pydantic>=2.12.5` - Schema validation for definitions, workloads, solutions, traces, contracts, and dataset manifests in `src/sol_execbench/core/data/` and `src/sol_execbench/core/dataset/`.
- `click>=8.0` and `rich>=13.0` - User-facing command-line interface in `src/sol_execbench/cli/`.
- `safetensors>=0.7.0` / locked `0.7.0` - Workload tensor blob loading in `src/sol_execbench/core/bench/io.py` and `src/sol_execbench/driver/templates/eval_driver.py`.
- `datasets>=4.8.2` / locked `4.8.2` - Downloading `nvidia/SOL-ExecBench` in `scripts/download_solexecbench.py`.
- `torch-c-dlpack-ext>=0.1.5` / locked `0.1.5` - Runtime dependency declared in `pyproject.toml` for DLPack extension support.
- `apache-tvm-ffi>=0.1.9` / locked `0.1.9` - Runtime dependency declared in `pyproject.toml`.

**Infrastructure:**
- ROCm host tools: `rocminfo`, `rocm-smi` or `amd-smi`, `hipcc`, `rocm_agent_enumerator`, and `rocprofv3`; documented in `docs/rocm.md` and invoked from `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/core/bench/clock_lock.py`, and `src/sol_execbench/core/bench/rocm_profiler.py`.
- ROCm native libraries: hipBLAS, MIOpen, Composable Kernel, and rocWMMA; dependency expectations are documented in `docs/rocm_libraries.md` and checked by `src/sol_execbench/core/diagnostics.py`.
- Docker GPU passthrough: `/dev/kfd`, `/dev/dri`, `video` group, unconfined seccomp, and host IPC in `docs/rocm.md` and `scripts/run_docker.sh`.
- Hugging Face CLI: `hf download` for `flashinfer-ai/flashinfer-trace` in `scripts/download_data.sh`.

## Configuration

**Environment:**
- No required application-level `.env` file is detected or documented; `docs/CONFIGURATION.md` states configuration is via CLI flags, optional benchmark `config.json`, package config, Docker env vars, and ROCm env vars.
- Container env vars are declared in `docker/Dockerfile`: `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, `PATH`, `LD_LIBRARY_PATH`, `UV_CACHE_DIR`, `UV_LINK_MODE`, `UV_COMPILE_BYTECODE`, `UV_PYTHON_DOWNLOADS`, `UV_PROJECT_ENVIRONMENT`, and `PYTHONPATH`.
- Runtime env vars include `FLASHINFER_TRACE_DIR` for safetensors lookup in `src/sol_execbench/driver/templates/eval_driver.py`, `SOL_EXECBENCH_CLOCKS_LOCKED` in `src/sol_execbench/core/bench/clock_lock.py`, `SOL_EXECBENCH_SCLK_LEVEL` and `SOL_EXECBENCH_MCLK_LEVEL` in `src/sol_execbench/core/bench/clock_lock.py`, and subprocess `PYTORCH_ALLOC_CONF=expandable_segments:True` in `src/sol_execbench/cli/main.py`.
- Native extension builds set `PYTORCH_ROCM_ARCH` from solution target hardware when absent in `src/sol_execbench/driver/templates/build_ext.py`.

**Build:**
- Python package metadata and dependency indexes: `pyproject.toml`.
- Locked dependencies: `uv.lock`.
- Docker image: `docker/Dockerfile`.
- Docker wrapper: `scripts/run_docker.sh`.
- Container entrypoint and clock lock lifecycle: `docker/entrypoint.sh`.
- Pre-commit lint and DCO hook configuration: `.pre-commit-config.yaml`.
- Benchmark config schema: `src/sol_execbench/core/bench/config/benchmark_config.py`.

## Platform Requirements

**Development:**
- Use Python `>=3.12,<3.14` with `uv sync --all-groups`.
- Use a Linux host with ROCm `>=7.0`, AMD GPU device access, and ROCm tools for GPU evaluation.
- Use native Linux Docker for container runs; Docker Desktop is rejected for ROCm passthrough per `docs/rocm.md` and `scripts/run_docker.sh`.
- Use `hipcc` and ROCm development headers for native HIP/C++ and library examples.
- Use `rocprofv3` for profiler-backed timing evidence in `src/sol_execbench/core/bench/rocm_profiler.py`.

**Production:**
- Not a hosted web service. The production-like target is a local or containerized ROCm evaluation environment running `sol-execbench` on AMD GPUs.
- Supported hardware metadata includes RDNA 4 (`gfx1200`) and CDNA 3 (`gfx940`, `gfx941`, `gfx942`) schema/build paths, with validation status documented in `docs/rocm.md`.
- Benchmark assets are local filesystem data under `data/SOL-ExecBench/benchmark` and `data/flashinfer-trace`.

---

*Stack analysis: 2026-05-24*
