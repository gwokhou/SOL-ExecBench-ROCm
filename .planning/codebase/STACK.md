# Technology Stack

**Analysis Date:** 2026-05-26

## Languages

**Primary:**
- Python `>=3.12,<3.14` - Package source in `src/sol_execbench/`, CLI entry points in `src/sol_execbench/cli/`, driver templates in `src/sol_execbench/driver/templates/`, scripts in `scripts/`, and tests in `tests/`.
- HIP/C++ - Native benchmark solution language accepted by the schema in `src/sol_execbench/core/data/solution.py`, staged and compiled through `src/sol_execbench/driver/templates/build_ext.py`, with examples in `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, and `examples/rocwmma/`.

**Secondary:**
- Bash - Docker and data helper scripts in `scripts/run_docker.sh`, `scripts/download_data.sh`, and `docker/entrypoint.sh`.
- JSON / JSONL - Public benchmark schemas and runtime artifacts in `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, and trace JSONL written by `src/sol_execbench/cli/main.py`.
- Markdown - User and researcher documentation in `README.md` and `docs/`.

## Runtime

**Environment:**
- Python `3.12` is the checked-in local version in `.python-version`; `pyproject.toml` allows `>=3.12,<3.14`.
- ROCm is the supported GPU runtime. `README.md` and `docs/GETTING-STARTED.md` state ROCm-capable AMD hardware and access to `/dev/kfd` and `/dev/dri` are required for GPU evaluation.
- Docker runtime uses `rocm/dev-ubuntu-24.04:7.1.1-complete` in `docker/Dockerfile`.
- The Docker image sets `ROCM_PATH=/opt/rocm`, `HIP_PATH=/opt/rocm`, `HIP_PLATFORM=amd`, `LD_LIBRARY_PATH=/opt/rocm/lib`, and a ROCm-aware `PATH` in `docker/Dockerfile`.

**Package Manager:**
- `uv` - Declared workflow in `README.md`, `docs/GETTING-STARTED.md`, and `pyproject.toml`.
- Docker image copies `uv` `0.5.11` from `ghcr.io/astral-sh/uv:0.5.11` in `docker/Dockerfile`.
- Lockfile: present at `uv.lock`.

## Frameworks

**Core:**
- PyTorch `2.10.0+rocm7.1` on Linux/Windows, `2.10.0` elsewhere - Tensor runtime, reference execution, HIP-backed `torch.cuda` device APIs, and native extension builds through `torch.utils.cpp_extension`; configured in `pyproject.toml` and `uv.lock`.
- torchvision `0.25.0+rocm7.1` on Linux/Windows, `0.25.0` elsewhere - Paired PyTorch dependency declared in `pyproject.toml` and `uv.lock`.
- Triton ROCm `3.6.0` on Linux - Triton solution runtime and examples under `examples/triton/`; declared as `triton-rocm` in `pyproject.toml`.
- Pydantic `2.12.5` - Public data models and diagnostic payloads in `src/sol_execbench/core/data/`, `src/sol_execbench/core/environment.py`, and `src/sol_execbench/core/toolchain.py`.
- Click `8.3.1` - CLI command definitions in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`.
- Rich `14.3.3` - CLI tables and progress output in `src/sol_execbench/cli/main.py`.

**Testing:**
- pytest `9.0.2` - Test runner configured by `[tool.pytest.ini_options]` in `pyproject.toml`.
- pytest-xdist `3.8.0` - Parallel test execution via `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.
- Test markers include `cpp`, `requires_rocm`, `requires_rdna4`, `requires_cdna3`, and legacy `requires_cutile` in `pyproject.toml`.

**Build/Dev:**
- Hatchling - Build backend declared in `[build-system]` in `pyproject.toml`.
- Ninja `1.13.0` - Native extension build dependency declared in `pyproject.toml`.
- Ruff `0.15.14` in `uv.lock` - Linting and formatting; repository exclusions and ignored rule `E741` are in `pyproject.toml`.
- Ty `0.0.39` - Type checking configured with `include = ["src", "tests"]` in `pyproject.toml`.
- pre-commit - `.pre-commit-config.yaml` runs Ruff hooks and a local DCO sign-off check.

## Key Dependencies

**Critical:**
- `torch` `2.10.0+rocm7.1` - Required for all GPU execution, HIP-backed device events, reference execution, and `torch.utils.cpp_extension` builds in `src/sol_execbench/driver/templates/build_ext.py`.
- `triton-rocm` `3.6.0` - Required for Triton ROCm examples and solution language support in `examples/triton/` and `src/sol_execbench/core/data/solution.py`.
- `pydantic` `2.12.5` - Defines stable schemas for `Definition`, `Workload`, `Solution`, `Trace`, environment diagnostics, and toolchain routing.
- `safetensors` `0.7.0` - Loads safetensors workload inputs via `src/sol_execbench/core/bench/io.py`; lookup roots are wired from `FLASHINFER_TRACE_DIR` in `src/sol_execbench/driver/templates/eval_driver.py`.
- `datasets` `4.8.2` - Downloads the public SOL-ExecBench dataset from Hugging Face in `scripts/download_solexecbench.py`.
- `click` `8.3.1` and `rich` `14.3.3` - Implement the user-facing CLI in `src/sol_execbench/cli/main.py`.

**Infrastructure:**
- ROCm command-line tools - `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `rocm-smi`, `amd-smi`, and `rocprofv3` are probed or used in `docker/Dockerfile`, `docker/entrypoint.sh`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/diagnostics.py`, and `src/sol_execbench/core/toolchain.py`.
- ROCm libraries - hipBLAS, MIOpen, Composable Kernel, and rocWMMA headers/libraries are modeled in `src/sol_execbench/core/diagnostics.py` and documented in `docs/rocm_libraries.md`.
- `torch-c-dlpack-ext` `0.1.5` and `apache-tvm-ffi` `0.1.9` - Runtime dependencies declared in `pyproject.toml` and locked in `uv.lock`.
- Docker - ROCm container support is implemented by `docker/Dockerfile`, `docker/entrypoint.sh`, and `scripts/run_docker.sh`.

## Configuration

**Environment:**
- No required application-level `.env` file is used; `docs/CONFIGURATION.md` states configuration is handled through CLI flags, optional benchmark config JSON, package metadata, and environment variables.
- Runtime variables include `FLASHINFER_TRACE_DIR`, `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, `SOL_EXECBENCH_MCLK_LEVEL`, `PYTORCH_ALLOC_CONF`, and `PYTORCH_ROCM_ARCH`; usage is documented in `docs/CONFIGURATION.md` and implemented in `docker/entrypoint.sh`, `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/cli/main.py`, and `src/sol_execbench/driver/templates/build_ext.py`.
- Docker wrapper variables include `IMAGE_NAME`, `IMAGE_TAG`, `SOL_EXECBENCH_GPU_CLK_MHZ`, and `SOL_EXECBENCH_DRAM_CLK_MHZ` in `scripts/run_docker.sh`.

**Build:**
- `pyproject.toml` defines package metadata, scripts, dependencies, dependency groups, pytest markers, Ruff settings, Ty settings, and uv package indexes.
- `uv.lock` pins the resolved dependency graph.
- `docker/Dockerfile` builds the ROCm evaluation image and installs all dependency groups with `uv sync --frozen --all-groups`.
- `.pre-commit-config.yaml` defines local commit-time lint/format/sign-off hooks.

## Platform Requirements

**Development:**
- Use Python `3.12` by default from `.python-version`.
- Run `uv sync --all-groups` from `README.md` and `docs/GETTING-STARTED.md`.
- For GPU behavior, use a ROCm-capable AMD GPU visible to PyTorch and accessible via `/dev/kfd` and `/dev/dri`.
- For native HIP/C++ examples, ensure ROCm compiler tooling such as `hipcc` and library development headers are available, or use `./scripts/run_docker.sh --build`.

**Production:**
- Not a hosted service. The production-like target is a local or containerized ROCm benchmark runner invoked through `uv run sol-execbench ...` or `./scripts/run_docker.sh -- ...`.
- The container target is native Linux Docker with ROCm device passthrough; `scripts/run_docker.sh` rejects Docker Desktop contexts because `/dev/kfd` and `/dev/dri` passthrough is required.

---

*Stack analysis: 2026-05-26*
