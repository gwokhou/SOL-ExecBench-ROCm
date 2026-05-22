# Technology Stack

**Analysis Date:** 2026-05-22

## Languages

**Primary:**
- Python >=3.12,<3.14 - Package implementation under `src/sol_execbench/`, CLI tools in `src/sol_execbench/cli/`, dataset scripts in `scripts/`, tests in `tests/`.
- HIP/C++ - Native GPU solution source format compiled from staged problem directories by `src/sol_execbench/driver/templates/build_ext.py`; examples live in `examples/hip_cpp/`.

**Secondary:**
- Bash - Docker/data wrappers in `scripts/run_docker.sh`, `scripts/download_data.sh`, and `docker/entrypoint.sh`.
- JSON/JSONL - Public benchmark schemas and runtime inputs use `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, and trace JSONL across `examples/`, `tests/sol_execbench/samples/`, and generated staging directories.

## Runtime

**Environment:**
- Python >=3.12,<3.14 from `pyproject.toml`.
- ROCm >=7.0 host/runtime target documented in `README.md` and `docs/rocm.md`.
- Container runtime base image: `rocm/dev-ubuntu-24.04:7.1.1-complete` in `docker/Dockerfile`.
- ROCm toolchain expected on PATH: `hipcc`, `rocminfo`, `rocprofv3`, and either `amd-smi` or `rocm-smi` in `docker/Dockerfile` and `tests/docker/dependencies/test_rocm_runtime.py`.
- GPU devices required for container execution: `/dev/kfd` and `/dev/dri` checked by `scripts/run_docker.sh`.

**Package Manager:**
- uv - Project dependency manager and command runner.
- Docker image copies uv `0.5.11` from `ghcr.io/astral-sh/uv:0.5.11` in `docker/Dockerfile`.
- Lockfile: present at `uv.lock`.

## Frameworks

**Core:**
- PyTorch `2.10.0+rocm7.1` on Linux/Windows - ROCm tensor runtime, HIP-backed `torch.cuda` APIs, extension loading, event timing, and reference execution; configured in `pyproject.toml`.
- PyTorch `2.10.0` on non-Linux/non-Windows - Non-ROCm fallback wheel declaration in `pyproject.toml`.
- torchvision `0.25.0+rocm7.1` on Linux/Windows - ROCm wheel source configured in `pyproject.toml`.
- Triton ROCm `3.6.0` - Triton kernel category support and examples under `examples/triton/`; configured as `triton-rocm==3.6.0` in `pyproject.toml`.
- Pydantic `2.12.5` - Strongly typed schema models under `src/sol_execbench/core/data/`.
- Click `8.3.1` - CLI commands in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`.
- Rich `14.3.3` - Console tables and progress UI in `src/sol_execbench/cli/main.py`.
- safetensors `0.7.0` - Benchmark input blob loading through `src/sol_execbench/core/bench/io.py` and safetensor roots in `src/sol_execbench/driver/templates/eval_driver.py`.
- Hugging Face datasets `4.8.2` - SOL ExecBench dataset download in `scripts/download_solexecbench.py`.

**Testing:**
- pytest `9.0.2` - Test runner configured in `pyproject.toml`.
- pytest-xdist `3.8.0` - Parallel test execution via `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.
- Docker dependency smoke tests - ROCm/PyTorch/HIP/library checks in `tests/docker/dependencies/`.

**Build/Dev:**
- hatchling - Build backend in `pyproject.toml`.
- Ruff - Lint/format configuration in `pyproject.toml`; excludes `data`, `examples`, generated caches, and build outputs.
- ninja `1.13.0` - Native extension build dependency configured in `pyproject.toml`.
- torch.utils.cpp_extension - HIP/C++ extension build path in `src/sol_execbench/driver/templates/build_ext.py`.
- Docker - ROCm evaluation environment in `docker/Dockerfile`, launched by `scripts/run_docker.sh`.

## Key Dependencies

**Critical:**
- `torch==2.10.0+rocm7.1` - Core ROCm tensor runtime, HIP extension integration, timing events, and GPU device discovery in `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/driver/templates/eval_driver.py`, and `src/sol_execbench/driver/templates/build_ext.py`.
- `triton-rocm==3.6.0` - Triton ROCm solution support used by `examples/triton/` and tested in `tests/docker/dependencies/test_triton_rocm.py`.
- `pydantic>=2.12.5` - Public schema validation for definitions, workloads, solutions, traces, and config in `src/sol_execbench/core/data/`.
- `safetensors>=0.7.0` - File-backed benchmark tensor inputs loaded from staging directories and `FLASHINFER_TRACE_DIR`.
- `datasets>=4.8.2` - Downloads `nvidia/SOL-ExecBench` in `scripts/download_solexecbench.py`.
- `click>=8.0` and `rich>=13.0` - User-facing CLI command parsing and console reporting in `src/sol_execbench/cli/`.

**Infrastructure:**
- `torch-c-dlpack-ext>=0.1.5` - Runtime dependency declared in `pyproject.toml` for DLPack extension support.
- `apache-tvm-ffi>=0.1.9` - Runtime dependency declared in `pyproject.toml`.
- ROCm command-line tools - `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `rocprofv3`, `rocm-smi`, and `amd-smi` are used or validated in `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/core/diagnostics.py`, `docker/Dockerfile`, and `tests/docker/dependencies/`.
- Native ROCm candidate libraries - Solution schema recognizes `hipblas`, `miopen`, `ck`, and `rocwmma` in `src/sol_execbench/core/data/solution.py`; support status is documented in `docs/rocm_libraries.md`.

## Configuration

**Environment:**
- Package metadata and dependency sources are configured in `pyproject.toml`.
- PyTorch ROCm wheels use explicit uv indexes `pytorch-rocm71` (`https://download.pytorch.org/whl/rocm7.1`) and `pytorch-rocm-root` (`https://download.pytorch.org/whl/`) in `pyproject.toml`.
- Runtime subprocesses set `PYTORCH_ALLOC_CONF=expandable_segments:True` in `src/sol_execbench/cli/main.py`.
- Docker image sets `ROCM_PATH=/opt/rocm`, `HIP_PATH=/opt/rocm`, `HIP_PLATFORM=amd`, ROCm PATH entries, and `LD_LIBRARY_PATH=/opt/rocm/lib` in `docker/Dockerfile`.
- Docker wrapper exports `FLASHINFER_TRACE_DIR`, `SOL_EXECBENCH_GPU_CLK_MHZ`, and `SOL_EXECBENCH_DRAM_CLK_MHZ` into containers in `scripts/run_docker.sh`.
- Clock locking uses `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, and `SOL_EXECBENCH_MCLK_LEVEL` in `docker/entrypoint.sh` and `src/sol_execbench/core/bench/clock_lock.py`.
- Evaluation loads safetensor blobs from the staging directory first, then `FLASHINFER_TRACE_DIR` in `src/sol_execbench/driver/templates/eval_driver.py`.

**Build:**
- `pyproject.toml` - Build system, package metadata, dependencies, pytest config, Ruff config, and uv indexes.
- `uv.lock` - Locked dependency graph.
- `docker/Dockerfile` - ROCm runtime image and frozen dependency install.
- `docker/entrypoint.sh` - Container startup, clock lock setup, cleanup trap, and FlashInfer mount warning.
- `scripts/run_docker.sh` - Host Docker context validation, image build, ROCm device passthrough, mount, and env forwarding.
- `.pre-commit-config.yaml` - Pre-commit hook configuration.

## Platform Requirements

**Development:**
- Python 3.12+ with uv.
- Linux ROCm host for real GPU evaluation.
- AMD GPU visible through PyTorch ROCm for `requires_rocm`, `requires_rdna4`, `requires_cdna3`, and timing tests.
- ROCm runtime/tools: `rocminfo`, `rocm-smi` or `amd-smi`, `hipcc`; `rocprofv3` for profiler readiness.
- Native Linux Docker daemon for container evaluation; `scripts/run_docker.sh` rejects Docker Desktop contexts because ROCm device passthrough requires host `/dev/kfd` and `/dev/dri`.
- Hugging Face CLI is needed by `scripts/download_data.sh` for `hf download flashinfer-ai/flashinfer-trace`.

**Production:**
- Not a hosted service. Deployment target is local or containerized benchmark execution on ROCm-capable AMD GPU machines.
- Container target mounts the repository at `/sol-execbench`, installs into `/venv`, and runs commands through `docker/entrypoint.sh`.
- Supported validation claim in docs covers RDNA 4 `gfx1200`; CDNA 3 `gfx940`, `gfx941`, and `gfx942` are schema/build targets pending real hardware validation in `docs/rocm.md`.

---

*Stack analysis: 2026-05-22*
