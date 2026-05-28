# Technology Stack

**Analysis Date:** 2026-05-28

## Languages

**Primary:**
- Python `>=3.12,<3.14` - package code in `src/sol_execbench/`, CLI entry points in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`, utility scripts in `scripts/`, tests in `tests/`.

**Secondary:**
- HIP/C++ - staged benchmark candidate builds from `src/sol_execbench/driver/problem_packager.py` using `src/sol_execbench/driver/templates/build_ext.py`; example sources live under `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, and `examples/rocwmma/`.
- Bash - Docker and dataset helper scripts in `scripts/run_docker.sh`, `scripts/download_data.sh`, and `docker/entrypoint.sh`.
- JSON / JSONL - benchmark schemas and runtime artifacts: `definition.json`, `workload.jsonl`, `solution.json`, trace JSONL outputs, Docker target metadata in `docker/rocm-targets.json`, and AMD hardware metadata in `src/sol_execbench/data/amd_hardware_models/gfx1200.json`.
- Markdown - user and developer documentation in `README.md` and `docs/`.

## Runtime

**Environment:**
- Python `>=3.12,<3.14`, declared in `pyproject.toml`.
- ROCm user-space target is ROCm `7.1.1` by default via `docker/Dockerfile` and `docker/rocm-targets.json`; the project constraints target ROCm `>=7.0`.
- Linux ROCm GPU runtime is the production path for benchmark execution; CPU-safe paths support schemas, docs, matrix guardrails, and non-GPU tests.

**Package Manager:**
- `uv` - dependency resolution, virtualenv management, command execution, and Docker install flow.
- Lockfile: present at `uv.lock`.
- Docker copies `uv` from `ghcr.io/astral-sh/uv:0.5.11` in `docker/Dockerfile`.

## Frameworks

**Core:**
- Click `>=8.0` - command-line interface in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`.
- Rich `>=13.0` - CLI tables, progress display, and stderr console output in `src/sol_execbench/cli/main.py`.
- Pydantic `>=2.12.5` - schema models in `src/sol_execbench/core/data/`, runtime evidence models in `src/sol_execbench/core/runtime_evidence.py`, dependency matrix models in `src/sol_execbench/core/dependency_matrix.py`, and Docker target models in `src/sol_execbench/core/docker_matrix.py`.
- PyTorch `2.10.0+rocm7.1` on Linux/Windows - tensor execution, HIP event timing, ROCm runtime checks, and native extension compilation; configured in `pyproject.toml`.
- Triton ROCm `3.6.0` on Linux - Triton solution execution and examples; configured in `pyproject.toml`.
- `torch.utils.cpp_extension` - HIP/C++ native extension build path in `src/sol_execbench/driver/templates/build_ext.py`.

**Testing:**
- Pytest `>=9.0.2` - test runner configured in `pyproject.toml` and documented in `docs/TESTING.md`.
- pytest-xdist `>=3.5` - full-suite parallelism configured with `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.
- Ruff `>=0.4` - linting and formatting, configured in `pyproject.toml`.
- Ty `>=0.0.39` - type checking configured in `pyproject.toml`.

**Build/Dev:**
- Hatchling - PEP 517 build backend declared in `pyproject.toml`.
- Ninja `>=1.13.0` - native extension build support declared in `pyproject.toml`.
- Docker - ROCm container workflow in `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`, and `scripts/run_docker.sh`.
- GitHub Actions - CPU-safe quality CI in `.github/workflows/code-quality.yml`.

## Key Dependencies

**Critical:**
- `torch==2.10.0+rocm7.1` on Linux/Windows - primary ROCm tensor runtime, HIP-backed timing, GPU availability checks, and native extension build support.
- `torchvision==0.25.0+rocm7.1` on Linux/Windows - pinned companion dependency for the PyTorch ROCm stack.
- `triton-rocm==3.6.0` on Linux - Triton ROCm candidate kernel support.
- `pydantic>=2.12.5` - public benchmark schemas, trace schemas, compatibility records, and evidence sidecars.
- `safetensors>=0.7.0` - workload tensor blob loading in `src/sol_execbench/core/bench/io.py`.
- `datasets>=4.8.2` - Hugging Face dataset acquisition in `scripts/download_solexecbench.py`.

**Infrastructure:**
- `click>=8.0` and `rich>=13.0` - CLI UX and output formatting.
- `ninja>=1.13.0` - extension compilation helper.
- `torch-c-dlpack-ext>=0.1.5` - DLPack extension dependency declared in `pyproject.toml`.
- `apache-tvm-ffi>=0.1.9` - TVM FFI dependency declared in `pyproject.toml`.
- ROCm command-line tools - `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, `rocm-smi`, `rocprofv3`, `rocprofv3-avail`, `llvm-objdump`, `roc-objdump`, and `readelf` are probed or routed by `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, and `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- ROCm libraries - hipBLAS, MIOpen, Composable Kernel, and rocWMMA readiness is checked in `src/sol_execbench/core/diagnostics.py` and documented in `docs/rocm_libraries.md`.

## Configuration

**Environment:**
- No application-level `.env` file is required; `docs/CONFIGURATION.md` states configuration is driven by CLI flags, benchmark config JSON, package metadata, Docker variables, and ROCm/PyTorch environment variables.
- Runtime flags are defined in `src/sol_execbench/cli/main.py`, including `--config`, `--compile-timeout`, `--timeout`, `--output`, `--json`, `--lock-clocks`, `--keep-staging`, `--profile`, and `--static-evidence`.
- Optional benchmark config JSON maps to `BenchmarkConfig` in `src/sol_execbench/core/bench/config/benchmark_config.py`.
- Docker/runtime environment variables include `IMAGE_NAME`, `IMAGE_TAG`, `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, `FLASHINFER_TRACE_DIR`, `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, `SOL_EXECBENCH_MCLK_LEVEL`, `PYTORCH_ALLOC_CONF`, `PYTORCH_ROCM_ARCH`, `SOLEXECBENCH_ENV_SNAPSHOT`, `SOLEXECBENCH_ENV_SNAPSHOT_PATH`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION`.
- `scripts/run_docker.sh` also accepts matrix and evidence variables such as `SOL_EXECBENCH_COMPATIBILITY_ENTRY`, `SOL_EXECBENCH_COMPATIBILITY_MATRIX`, `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES`, and `SOL_EXECBENCH_DEPENDENCY_*` overrides for diagnostics.

**Build:**
- `pyproject.toml` defines package metadata, console scripts, dependencies, uv indexes, pytest markers, Ruff config, and Ty source includes.
- `uv.lock` pins resolved dependencies.
- `docker/Dockerfile` builds the ROCm development image, installs the locked `uv` environment, and exposes the package CLI.
- `docker/rocm-targets.json` declares supported container ROCm targets and PyTorch ROCm dependency policies.
- `.github/workflows/code-quality.yml` installs with `uv sync --locked --all-groups`, then runs Ruff, Ty, and CPU-safe pytest subsets.

## Platform Requirements

**Development:**
- Python `3.12` or `3.13` as validated in `.github/workflows/code-quality.yml`.
- `uv sync --all-groups` installs runtime and dev dependencies.
- CPU-safe checks use `uv run ruff check .`, `uv run ty check`, and pytest subsets under `tests/sol_execbench/` and `tests/examples/`.
- ROCm development checks require Linux, ROCm drivers/tools, PyTorch ROCm wheels, and access to `/dev/kfd` and `/dev/dri`.

**Production:**
- Local benchmark execution target is a ROCm-capable Linux host or the Docker environment launched by `./scripts/run_docker.sh`.
- Default container image is `rocm/dev-ubuntu-24.04:7.1.1-complete` from `docker/Dockerfile` and `docker/rocm-targets.json`.
- GPU evaluation requires AMD hardware visible to PyTorch ROCm, with project target architectures including RDNA 4 `gfx1200` and CDNA 3 `gfx940`/`gfx941`/`gfx942`.
- Optional clock locking requires `rocm-smi` and sudo permissions configured by `docker/Dockerfile` and used by `docker/entrypoint.sh` plus `src/sol_execbench/core/bench/clock_lock.py`.

---

*Stack analysis: 2026-05-28*
