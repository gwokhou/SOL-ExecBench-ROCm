# Technology Stack

**Analysis Date:** 2026-06-01

## Languages

**Primary:**
- Python 3.12+ / <3.14 - Package implementation, CLIs, schemas, scoring, reporting, dataset tooling, and tests under `src/sol_execbench/`, `scripts/`, and `tests/`; version floor is declared in `pyproject.toml`, and `.python-version` pins local development to `3.12`.

**Secondary:**
- Bash - Docker wrapper and dataset helper scripts in `scripts/run_docker.sh`, `scripts/download_data.sh`, and `docker/entrypoint.sh`.
- HIP/C++ - Native ROCm solution examples and staged compilation targets in `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, `examples/rocwmma/`, and `src/sol_execbench/driver/templates/build_ext.py`.
- JSON / JSONL - Benchmark schemas, workloads, Docker target manifests, sidecars, and examples in `examples/**/definition.json`, `examples/**/workload.jsonl`, `docker/rocm-targets.json`, and `docs/examples/`.
- Markdown - User and engineering documentation in `README.md`, `docs/`, `AGENTS.md`, `CONTRIBUTING.md`, and `SECURITY.md`.

## Runtime

**Environment:**
- CPython `>=3.12,<3.14` from `pyproject.toml`.
- ROCm runtime baseline is ROCm 7.x; Docker defaults to `rocm/dev-ubuntu-24.04:7.1.1-complete` in `docker/Dockerfile` and `docker/rocm-targets.json`.
- GPU runtime expects AMD ROCm device nodes `/dev/kfd` and `/dev/dri`; Docker wrapper mounts them in `scripts/run_docker.sh`.
- PyTorch uses the ROCm compatibility namespace (`torch.cuda`) for HIP-backed devices in `src/sol_execbench/driver/templates/eval_driver.py` and `src/sol_execbench/core/bench/eval_runtime.py`.

**Package Manager:**
- `uv` - Project dependency and virtualenv manager; `uv.lock` is present.
- Docker copies `uv` `0.5.11` from `ghcr.io/astral-sh/uv:0.5.11` in `docker/Dockerfile`.
- Lockfile: present at `uv.lock`.

## Frameworks

**Core:**
- Click `>=8.0` / locked `8.3.1` - CLI command handling in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`.
- Rich `>=13.0` / locked `14.3.3` - CLI tables, status output, and progress rendering in `src/sol_execbench/cli/main.py`.
- Pydantic `>=2.12.5` / locked `2.12.5` - Strict schema models under `src/sol_execbench/core/data/`, compatibility matrix models in `src/sol_execbench/core/compatibility.py`, runtime evidence models in `src/sol_execbench/core/runtime_evidence.py`, and Docker target models in `src/sol_execbench/core/docker_matrix.py`.
- PyTorch `2.10.0+rocm7.1` on Linux/Windows ROCm targets - Tensor execution, HIP-backed timing, extension builds, and device probing in `src/sol_execbench/core/bench/`, `src/sol_execbench/driver/templates/`, and examples.
- Triton ROCm `3.6.0` on Linux - Triton solution category used by examples under `examples/triton/` and dependency tests in `tests/docker/dependencies/test_triton_rocm.py`.

**Testing:**
- Pytest `>=9.0.2` / locked `9.0.2` - Test runner configured in `pyproject.toml`; tests live under `tests/`.
- pytest-xdist `>=3.5` / locked `3.8.0` - Parallel pytest execution via `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.

**Build/Dev:**
- Hatchling - Build backend declared in `pyproject.toml`.
- Ruff `>=0.4` / locked `0.15.14` - Formatting and linting configured in `pyproject.toml` and `.pre-commit-config.yaml`.
- Ty `>=0.0.39` / locked `0.0.39` - Type checking configured in `pyproject.toml` and CI in `.github/workflows/code-quality.yml`.
- Ninja `>=1.13.0` / locked `1.13.0` - Native extension build helper dependency for HIP/C++ compilation.
- `torch.utils.cpp_extension` - Runtime native build bridge in `src/sol_execbench/driver/templates/build_ext.py`.
- Docker - Reproducible ROCm development/evaluation environment in `docker/Dockerfile` and `scripts/run_docker.sh`.
- Pre-commit - Ruff hooks and DCO commit-message hook in `.pre-commit-config.yaml`.

## Key Dependencies

**Critical:**
- `torch==2.10.0+rocm7.1` on ROCm platforms - Core execution engine, HIP device events, tensor generation, native extension loader, and GPU probing.
- `torchvision==0.25.0+rocm7.1` on ROCm platforms - Locked alongside PyTorch ROCm wheels in `pyproject.toml`, `uv.lock`, `docker/Dockerfile`, and `docker/rocm-targets.json`.
- `triton-rocm==3.6.0` on Linux - ROCm Triton kernel support and Triton example execution.
- `pydantic>=2.12.5` - Public schemas for definitions, workloads, solutions, traces, contracts, Matrix evidence, and report sidecars.
- `click>=8.0` and `rich>=13.0` - User-facing CLIs and formatted terminal output.
- `datasets>=4.8.2` - Hugging Face dataset access in `scripts/download_solexecbench.py`.
- `safetensors>=0.7.0` - External tensor blob loading for workloads in `src/sol_execbench/core/bench/io.py`.

**Infrastructure:**
- `ninja>=1.13.0` - Native extension build acceleration for HIP/C++ solutions.
- `torch-c-dlpack-ext>=0.1.5` - Runtime dependency declared in `pyproject.toml` and locked in `uv.lock` for DLPack-related extension support.
- `apache-tvm-ffi>=0.1.9` - Runtime dependency declared in `pyproject.toml` and locked in `uv.lock`.
- ROCm CLI tools `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, `rocm-smi`, `rocprofv3`, `rocprofv3-avail` - Toolchain, diagnostics, profiling, and clock control in `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and `src/sol_execbench/core/toolchain.py`.
- Static binary tools `readelf`, `llvm-objdump`, `roc-objdump`, and `rga` - Toolchain routing and static evidence support in `src/sol_execbench/core/toolchain.py` and `src/sol_execbench/core/bench/static_kernel_evidence.py`.

## Configuration

**Environment:**
- No required application-level `.env` file is present or required; `docs/CONFIGURATION.md` states normal host CLI startup has no required env vars.
- Runtime and diagnostic env vars are optional and source-backed: `SOLEXECBENCH_ENV_SNAPSHOT`, `SOLEXECBENCH_ENV_SNAPSHOT_PATH`, `PYTORCH_ALLOC_CONF`, `PYTORCH_ROCM_ARCH`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, `SOL_EXECBENCH_MCLK_LEVEL`, and `FLASHINFER_TRACE_DIR`.
- Docker wrapper env/config surface includes `IMAGE_NAME`, `IMAGE_TAG`, `ROCM_DOCKER_IMAGE`, `ROCM_DOCKER_TAG`, `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES`, `SOL_EXECBENCH_ALLOW_UNTESTED_TARGET_SMOKE`, `SOL_EXECBENCH_RECORD_CONTAINER_VALIDATION`, `SOL_EXECBENCH_COMPATIBILITY_ENTRY`, and `SOL_EXECBENCH_COMPATIBILITY_MATRIX` in `scripts/run_docker.sh`.
- Benchmark config is optional JSON loaded by `src/sol_execbench/core/bench/config/benchmark_config.py` with defaults: `warmup_runs=10`, `iterations=50`, `lock_clocks=False`, `benchmark_reference=True`, `seed=200`.

**Build:**
- `pyproject.toml` defines package metadata, dependencies, CLI scripts, pytest markers, Ruff settings, Ty roots, and UV indexes.
- `uv.lock` pins dependency resolution.
- `.python-version` pins local Python to `3.12`.
- `.pre-commit-config.yaml` configures Ruff and DCO sign-off enforcement.
- `docker/Dockerfile` defines the ROCm/PyTorch/Triton container image.
- `docker/rocm-targets.json` declares supported ROCm Docker target stacks for ROCm 7.0.2, 7.1.1, and 7.2.0.
- `.github/workflows/code-quality.yml` runs GitHub Actions quality checks on Python 3.12 and 3.13.

## Platform Requirements

**Development:**
- Python 3.12 or 3.13 with `uv sync --all-groups`.
- ROCm-capable AMD hardware for GPU behavior; CPU-safe tests can run without GPU.
- ROCm user-space tools on PATH for full diagnostics and native execution: `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `rocprofv3`, and either `amd-smi` or `rocm-smi`.
- Docker with access to `/dev/kfd` and `/dev/dri` for containerized evaluation through `scripts/run_docker.sh`.
- Hugging Face CLI is optional for `scripts/download_data.sh`; `scripts/download_solexecbench.py` uses the `datasets` Python package directly.

**Production:**
- Not a hosted web service. The supported runtime target is local or Dockerized benchmark execution on AMD ROCm systems.
- Reproducible container targets are declared in `docker/rocm-targets.json` and built by `scripts/run_docker.sh`.

---

*Stack analysis: 2026-06-01*
