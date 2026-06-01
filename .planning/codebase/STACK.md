# Technology Stack

**Analysis Date:** 2026-06-01

## Languages

**Primary:**
- Python `>=3.12,<3.14` - package source in `src/sol_execbench/`, CLI scripts in `src/sol_execbench/cli/`, dataset/report scripts in `scripts/`, and tests in `tests/`; configured in `pyproject.toml` and `.python-version`.
- HIP/C++ - native solution categories and examples in `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, and `examples/rocwmma/`; build staging is in `src/sol_execbench/driver/problem_packager.py` and `src/sol_execbench/driver/templates/build_ext.py`.

**Secondary:**
- Bash - Docker and dataset helpers in `scripts/run_docker.sh`, `scripts/download_data.sh`, and `docker/entrypoint.sh`.
- JSON/JSONL - benchmark contracts and examples in `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, `docs/trace.md`, `examples/*/*/*.json`, and `examples/*/*/*.jsonl`.
- Markdown - user and research documentation in `README.md` and `docs/`.

## Runtime

**Environment:**
- Python `3.12` - local version pin in `.python-version`; package range in `pyproject.toml`.
- ROCm user-space `7.x` - development target in `README.md`; Docker target matrix declares ROCm `7.0.2`, `7.1.1`, and `7.2.0` in `docker/rocm-targets.json`.
- Default Docker runtime image `rocm/dev-ubuntu-24.04:7.1.1-complete` - selected by `docker/Dockerfile` and `docker/rocm-targets.json`.
- AMD GPU runtime access through `/dev/kfd` and `/dev/dri` - required by `README.md` and passed by `scripts/run_docker.sh`.

**Package Manager:**
- `uv` - dependency resolution and command execution in `pyproject.toml`, `uv.lock`, `README.md`, `docs/CONFIGURATION.md`, and `docker/Dockerfile`.
- Lockfile: present at `uv.lock`.

## Frameworks

**Core:**
- PyTorch `2.10.0+rocm7.1` on Linux/Windows, `2.10.0` elsewhere - tensor runtime, ROCm device execution, HIP-backed `torch.cuda` compatibility APIs, and extension builds; configured in `pyproject.toml` and `docker/rocm-targets.json`.
- torchvision `0.25.0+rocm7.1` on Linux/Windows, `0.25.0` elsewhere - paired PyTorch dependency configured in `pyproject.toml` and `docker/rocm-targets.json`.
- Triton ROCm `3.6.0` on Linux - Triton solution category and profiler timing policies; configured in `pyproject.toml`, `docker/Dockerfile`, and `docker/rocm-targets.json`.
- Pydantic `>=2.12.5` - schema models for definitions, workloads, solutions, traces, diagnostics, and compatibility artifacts in `src/sol_execbench/core/data/`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, and `src/sol_execbench/core/compatibility.py`.
- Click `>=8.0` - CLI command surface in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`.
- Rich `>=13.0` - terminal rendering for CLI tables and progress in `src/sol_execbench/cli/main.py`.

**Testing:**
- pytest `>=9.0.2` - test runner configured in `pyproject.toml`; test suite under `tests/`.
- pytest-xdist `>=3.5` - parallel test execution via `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`; serial grouping appears in GPU-sensitive tests under `tests/`.

**Build/Dev:**
- Hatchling - Python build backend in `pyproject.toml`.
- Ruff `>=0.4` - linting and formatting configuration in `pyproject.toml`; generated data and examples are excluded.
- Ty `>=0.0.39` - type checking for `src` and `tests` configured in `pyproject.toml`.
- Ninja `>=1.13.0` - native extension build helper dependency in `pyproject.toml`.
- Docker - ROCm container workflow in `docker/Dockerfile` and `scripts/run_docker.sh`.
- `torch.utils.cpp_extension` - HIP/C++ extension build backend in `src/sol_execbench/driver/templates/build_ext.py`.

## Key Dependencies

**Critical:**
- `torch` / PyTorch ROCm - canonical tensor runtime, reference execution, device events, HIP-backed stream APIs, and native extension loading; used across `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/timing.py`, and `src/sol_execbench/driver/templates/build_ext.py`.
- `triton-rocm` - supported solution language category `triton` in `src/sol_execbench/core/data/solution.py` and examples under `examples/triton/`.
- `pydantic` - strict contract validation for public benchmark schemas in `src/sol_execbench/core/data/`.
- `datasets` - Hugging Face dataset acquisition in `scripts/download_solexecbench.py`.
- `safetensors` - FlashInfer trace and benchmark asset loading support noted in `pyproject.toml`, `README.md`, and `docs/compliance.md`.
- `click` and `rich` - primary user-facing CLI stack in `src/sol_execbench/cli/main.py`.

**Infrastructure:**
- `torch-c-dlpack-ext>=0.1.5` - DLPack/native tensor interoperability dependency declared in `pyproject.toml`.
- `apache-tvm-ffi>=0.1.9` - FFI infrastructure dependency declared in `pyproject.toml`.
- ROCm command-line tools `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, `rocm-smi`, `rocprofv3`, `rocprofv3-avail`, `llvm-objdump`, and `readelf` - probed or routed by `docker/Dockerfile`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/bench/clock_lock.py`, and `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- ROCm libraries hipBLAS, MIOpen, Composable Kernel, and rocWMMA - supported solution categories in `src/sol_execbench/core/data/solution.py` and example directories under `examples/`.

## Configuration

**Environment:**
- No required `.env` file is present or documented; `docs/CONFIGURATION.md` states there is no required application-level `.env`.
- Runtime environment variables are optional and grouped by purpose in `docs/CONFIGURATION.md`.
- Device visibility and runtime evidence use `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, `CUDA_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL`; collected in `src/sol_execbench/core/environment.py` and forwarded by `scripts/run_docker.sh`.
- Benchmark subprocesses set `PYTORCH_ALLOC_CONF=expandable_segments:True` in `src/sol_execbench/cli/main.py`.
- Native builds use `PYTORCH_ROCM_ARCH` when already set, or derive it from solution hardware targets in `src/sol_execbench/driver/templates/build_ext.py`.
- Clock locking uses `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, and `SOL_EXECBENCH_MCLK_LEVEL` in `src/sol_execbench/core/bench/clock_lock.py` and `docker/entrypoint.sh`.
- Diagnostic sidecars use `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` in `src/sol_execbench/cli/main.py`.
- Docker wrapper and compatibility evidence use `SOL_EXECBENCH_*`, `ROCM_DOCKER_IMAGE`, `ROCM_DOCKER_TAG`, `IMAGE_NAME`, and `IMAGE_TAG` in `scripts/run_docker.sh` and `docs/CONFIGURATION.md`.

**Build:**
- `pyproject.toml` defines package metadata, dependencies, console scripts, pytest markers, Ruff settings, Ty roots, and uv package indexes.
- `uv.lock` is the committed resolver lockfile.
- `docker/Dockerfile` builds the default ROCm image, installs uv, syncs dependencies, and overlays target-specific PyTorch/Triton ROCm wheels.
- `docker/rocm-targets.json` declares supported Docker target IDs, ROCm versions, Docker tags, PyTorch ROCm wheel policies, and Triton ROCm wheel policy.
- `src/sol_execbench/core/bench/config/benchmark_config.py` defines runtime benchmark config fields loaded from optional JSON via `--config`.

## Platform Requirements

**Development:**
- Python `>=3.12,<3.14`, uv, and repository dependencies from `pyproject.toml`.
- Linux with ROCm-capable AMD hardware for GPU evaluation, `/dev/kfd` and `/dev/dri` access, and ROCm 7.x user-space tooling as described in `README.md`.
- ROCm HIP development headers and libraries for native `hip_cpp`, `hipblas`, `miopen`, `ck`, and `rocwmma` paths; availability-sensitive tests are marked in `tests/conftest.py`.
- Docker for containerized GPU workflows through `./scripts/run_docker.sh`.

**Production:**
- Not a hosted service. The deployment target is a local or containerized ROCm benchmark environment running the `sol-execbench` and `sol-execbench-baseline` console scripts.
- Containerized execution uses `rocm/dev-ubuntu-24.04` images selected by `docker/rocm-targets.json`, with AMD GPU devices mounted by `scripts/run_docker.sh`.

---

*Stack analysis: 2026-06-01*
