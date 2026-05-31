# Technology Stack

**Analysis Date:** 2026-05-31

## Languages

**Primary:**
- Python `>=3.12,<3.14` - Package implementation, CLIs, data schemas, benchmark orchestration, dataset scripts, and tests in `src/sol_execbench/`, `scripts/`, and `tests/`.

**Secondary:**
- HIP/C++ - User solution sources for native ROCm categories and staged extension builds through `src/sol_execbench/driver/problem_packager.py` and `src/sol_execbench/driver/templates/build_ext.py`.
- Bash - Docker wrapper, dataset helper, and container entrypoint in `scripts/run_docker.sh`, `scripts/download_data.sh`, and `docker/entrypoint.sh`.
- JSON/JSONL - Public benchmark schemas, traces, matrix metadata, Docker target metadata, sample problem definitions, and workload data in `src/sol_execbench/core/data/`, `docker/rocm-targets.json`, `examples/`, and `tests/sol_execbench/samples/`.
- Markdown - User and developer documentation in `README.md` and `docs/`.

## Runtime

**Environment:**
- Python `3.12` local default from `.python-version`; CI also tests Python `3.13` in `.github/workflows/code-quality.yml`.
- ROCm-capable Linux runtime for GPU execution, with `/dev/kfd` and `/dev/dri` access documented in `README.md` and probed by `scripts/run_docker.sh`.
- ROCm user-space `7.x`; default Docker target is ROCm `7.1.1` from `docker/rocm-targets.json` and `docker/Dockerfile`.
- PyTorch exposes ROCm devices through CUDA-compatible APIs; benchmark code uses `torch.cuda` with ROCm/HIP backing in `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/correctness.py`, and `src/sol_execbench/driver/templates/eval_driver.py`.

**Package Manager:**
- `uv` - Dependency management, lockfile execution, Docker installation, and CLI/test commands.
- Lockfile: present as `uv.lock`.
- Build backend: Hatchling via `[build-system]` in `pyproject.toml`.

## Frameworks

**Core:**
- Click `>=8.0` - CLI commands and options in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`.
- Rich `>=13.0` - Terminal tables/progress output in `src/sol_execbench/cli/main.py`.
- Pydantic `>=2.12.5` - Strict schemas and model validation in `src/sol_execbench/core/data/`, `src/sol_execbench/core/compatibility.py`, `src/sol_execbench/core/runtime_evidence.py`, and report models.
- PyTorch `2.10.0+rocm7.1` on Linux/Windows by default - Tensor runtime, reference implementations, timing events, and HIP/C++ extension build integration.
- Torchvision `0.25.0+rocm7.1` on Linux/Windows by default - Declared runtime dependency and dependency matrix evidence.
- Triton ROCm `3.6.0` on Linux - Triton solution category and container dependency policy in `pyproject.toml` and `docker/rocm-targets.json`.
- `torch.utils.cpp_extension` - Native HIP/C++ extension build path in `src/sol_execbench/driver/templates/build_ext.py`.

**Testing:**
- Pytest `>=9.0.2` - Test runner configured under `[tool.pytest.ini_options]` in `pyproject.toml`.
- pytest-xdist `>=3.5` - Parallel test execution via `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.
- Click `CliRunner` - CLI tests under `tests/sol_execbench/`.
- Hardware markers: `cpp`, `requires_rocm`, `requires_rdna4`, `requires_cdna3`, and legacy `requires_cutile` are declared in `pyproject.toml`.

**Build/Dev:**
- Ruff `>=0.4` - Lint and format tool configured in `pyproject.toml`.
- Ty `>=0.0.39` - Static type checking configured with `[tool.ty.src]` in `pyproject.toml`.
- Ninja `>=1.13.0` - Native extension build dependency from `pyproject.toml`.
- Docker - ROCm evaluation container defined by `docker/Dockerfile` and launched by `scripts/run_docker.sh`.
- GitHub Actions - CPU-safe quality pipeline in `.github/workflows/code-quality.yml`.

## Key Dependencies

**Critical:**
- `torch==2.10.0+rocm7.1` - Core tensor runtime, ROCm device access, HIP-backed timing, and extension build backend.
- `torchvision==0.25.0+rocm7.1` - Tracked as part of the PyTorch ROCm dependency policy in `pyproject.toml`, `docker/rocm-targets.json`, and `src/sol_execbench/core/dependency_matrix.py`.
- `triton-rocm==3.6.0` - Linux Triton ROCm solution support and dependency matrix evidence.
- `pydantic>=2.12.5` - Public schema validation for definitions, workloads, solutions, traces, runtime evidence, compatibility matrices, and reports.
- `datasets>=4.8.2` - Hugging Face SOL-ExecBench dataset acquisition in `scripts/download_solexecbench.py`.
- `safetensors>=0.7.0` - FlashInfer trace and tensor artifact support referenced by evaluation paths and `scripts/download_data.sh`.
- `torch-c-dlpack-ext>=0.1.5` and `apache-tvm-ffi>=0.1.9` - Runtime dependencies for tensor interop / TVM FFI compatibility paths declared in `pyproject.toml`.

**Infrastructure:**
- `click>=8.0` - CLI parsing and error handling in `src/sol_execbench/cli/`.
- `rich>=13.0` - Human-readable CLI reporting in `src/sol_execbench/cli/main.py`.
- `ninja>=1.13.0` - Native build acceleration for PyTorch extension builds.
- ROCm command-line tools - `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `rocprofv3`, `rocprofv3-avail`, `rocm-smi`/`amd-smi`, `llvm-objdump`, and `readelf` are probed or routed by `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/bench/clock_lock.py`, and `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- Docker image source - `rocm/dev-ubuntu-24.04` tags declared in `docker/Dockerfile` and `docker/rocm-targets.json`.
- `ghcr.io/astral-sh/uv:0.5.11` - Dockerfile source for the `uv` binary in `docker/Dockerfile`.

## Configuration

**Environment:**
- No required application `.env` file is detected; environment configuration is optional and documented in `docs/CONFIGURATION.md`.
- CLI environment sidecars are controlled by `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` in `src/sol_execbench/cli/main.py`.
- ROCm visibility variables recorded by evidence paths include `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, `CUDA_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL` in `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/runtime_evidence.py`, and `docs/CONFIGURATION.md`.
- Clock locking uses `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, and `SOL_EXECBENCH_MCLK_LEVEL` in `src/sol_execbench/core/bench/clock_lock.py` and `docker/entrypoint.sh`.
- Docker wrapper diagnostics and compatibility sidecars use `SOL_EXECBENCH_*` variables listed in `docs/CONFIGURATION.md` and consumed by `scripts/run_docker.sh`.
- CLI subprocesses set `PYTORCH_ALLOC_CONF=expandable_segments:True` for staged compile/evaluation in `src/sol_execbench/cli/main.py`.

**Build:**
- `pyproject.toml` declares package metadata, scripts, dependencies, uv indexes, pytest markers, Ruff exclusions, and Ty source roots.
- `uv.lock` pins dependency resolution.
- `.python-version` pins local Python to `3.12`.
- `docker/Dockerfile` defines the ROCm development image, Python dependency install, and runtime env.
- `docker/rocm-targets.json` declares supported ROCm container targets and PyTorch/Triton wheel policies.
- `.github/workflows/code-quality.yml` runs `uv sync --locked --all-groups`, Ruff, Ty, and CPU-safe pytest jobs for Python `3.12` and `3.13`.
- `docs/CONFIGURATION.md` is the canonical configuration reference for CLI flags, JSON config shape, Docker target metadata, and environment variables.

## Platform Requirements

**Development:**
- Python `>=3.12,<3.14`.
- `uv` available on the host.
- ROCm-capable AMD hardware for GPU evaluation and GPU-marked tests.
- ROCm tools on PATH for native host execution: at least `hipcc`, `rocminfo`, `rocm_agent_enumerator`, and optionally `rocprofv3`, `rocm-smi`/`amd-smi`, `llvm-objdump`, and `readelf`.
- Device node access to `/dev/kfd` and `/dev/dri` for ROCm GPU execution.
- Docker for the container workflow in `scripts/run_docker.sh`.
- Hugging Face CLI is optional for `scripts/download_data.sh`; the Python `datasets` package is used by `scripts/download_solexecbench.py`.

**Production:**
- Not a hosted web service; production-like usage is local or containerized benchmark execution.
- Primary deployment artifact is the Python package with console scripts `sol-execbench` and `sol-execbench-baseline` from `pyproject.toml`.
- Reproducible GPU evaluation target is the ROCm Docker workflow from `docker/Dockerfile`, `docker/rocm-targets.json`, and `scripts/run_docker.sh`.
- Supported ROCm container targets are `rocm-7.0.2-ubuntu-24.04-container`, `rocm-7.1.1-ubuntu-24.04-container`, and `rocm-7.2.0-ubuntu-24.04-container` in `docker/rocm-targets.json`.

---

*Stack analysis: 2026-05-31*
