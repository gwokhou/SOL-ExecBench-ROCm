# Technology Stack

**Analysis Date:** 2026-06-01

## Languages

**Primary:**
- Python 3.12+ - Package source, CLI, benchmark orchestration, schemas, scoring, dataset tooling, and tests under `src/sol_execbench/`, `scripts/`, and `tests/`; constrained by `requires-python = ">=3.12,<3.14"` in `pyproject.toml` and `.python-version`.

**Secondary:**
- HIP/C++ - Native benchmark solution examples and staged extension builds in `examples/hip_cpp/`, `examples/hipblas/`, `examples/ck/`, `examples/rocwmma/`, `src/sol_execbench/driver/templates/build_ext.py`, and `src/sol_execbench/driver/problem_packager.py`.
- Bash - Docker wrapper and container entrypoint in `scripts/run_docker.sh`, `scripts/download_data.sh`, and `docker/entrypoint.sh`.
- JSON/JSONL - Public benchmark schemas and runtime artifacts in `examples/*/*/definition.json`, `examples/*/*/solution_*.json`, `examples/*/*/workload.jsonl`, `docker/rocm-targets.json`, and trace/report outputs produced by `src/sol_execbench/cli/main.py`.

## Runtime

**Environment:**
- Python 3.12 local runtime from `.python-version`; package metadata allows Python 3.12 and 3.13 through `pyproject.toml`.
- ROCm Linux runtime is the supported GPU execution environment. Docker defaults to `rocm/dev-ubuntu-24.04:7.1.1-complete` in `docker/Dockerfile` and `docker/rocm-targets.json`.
- AMD GPU device access expects `/dev/kfd` and `/dev/dri`, forwarded by `scripts/run_docker.sh`.

**Package Manager:**
- uv - Used for dependency sync, lock management, Docker installs, and host wrapper helper commands in `pyproject.toml`, `uv.lock`, `docker/Dockerfile`, and `scripts/run_docker.sh`.
- Lockfile: present at `uv.lock`.

## Frameworks

**Core:**
- Click >=8.0 - CLI command definitions in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`.
- Rich >=13.0 - CLI progress/table rendering in `src/sol_execbench/cli/main.py`.
- Pydantic >=2.12.5 - Strict data models, trace schemas, compatibility matrices, and reports across `src/sol_execbench/core/data/`, `src/sol_execbench/core/compatibility.py`, `src/sol_execbench/core/dependency_matrix.py`, and `src/sol_execbench/core/environment.py`.
- PyTorch ROCm 2.10.0+rocm7.1 by default - Reference execution, tensor work, HIP-backed timing, and native extension builds in `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/`, and `src/sol_execbench/driver/templates/build_ext.py`.
- Triton ROCm 3.6.0 - Supported solution category and ROCm kernel path, configured in `pyproject.toml`, `docker/Dockerfile`, and `docker/rocm-targets.json`.

**Testing:**
- pytest >=9.0.2 - Test runner configured in `[tool.pytest.ini_options]` in `pyproject.toml`.
- pytest-xdist >=3.5 - Parallel test execution configured with `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.

**Build/Dev:**
- Hatchling - Python build backend configured under `[build-system]` in `pyproject.toml`.
- Ruff >=0.4 - Linting and formatting configured under `[tool.ruff]` and `[tool.ruff.lint]` in `pyproject.toml`.
- Ty >=0.0.39 - Type checking configured under `[tool.ty.src]` in `pyproject.toml`.
- Ninja >=1.13.0 - Native extension build dependency declared in `pyproject.toml`.
- Docker - GPU container build/run workflow in `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`, and `scripts/run_docker.sh`.

## Key Dependencies

**Critical:**
- torch 2.10.0 / 2.10.0+rocm7.1 - Core tensor runtime, ROCm device timing, and HIP/C++ extension bridge; declared in `pyproject.toml` and overridden per target in `docker/rocm-targets.json`.
- torchvision 0.25.0 / 0.25.0+rocm7.1 - Paired PyTorch ROCm dependency declared in `pyproject.toml` and Docker target policies.
- triton-rocm 3.6.0 - Linux-only Triton ROCm kernel support from PyTorch ROCm wheel indexes in `pyproject.toml`.
- pydantic >=2.12.5 - Public model validation and serialized contract stability in `src/sol_execbench/core/data/`.
- safetensors >=0.7.0 - Workload blob loading through `src/sol_execbench/core/bench/io.py` and dataset readiness checks in `src/sol_execbench/core/dataset/readiness.py`.
- datasets >=4.8.2 - Hugging Face dataset acquisition in `scripts/download_solexecbench.py`.
- torch-c-dlpack-ext >=0.1.5 and apache-tvm-ffi >=0.1.9 - Runtime/dependency matrix components declared in `pyproject.toml`.

**Infrastructure:**
- ROCm tools: `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, `rocm-smi`, `rocprofv3`, `llvm-objdump`, and `readelf` are probed or routed by `docker/Dockerfile`, `docker/entrypoint.sh`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- ROCm libraries: hipBLAS, MIOpen, Composable Kernel, and rocWMMA readiness is modeled in `src/sol_execbench/core/diagnostics.py` and documented by examples under `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, and `examples/rocwmma/`.

## Configuration

**Environment:**
- Normal host CLI startup has no required `.env` file; `docs/CONFIGURATION.md` states the repository does not contain a required application-level `.env`.
- CLI runtime configuration is primarily flags on `sol-execbench` in `src/sol_execbench/cli/main.py`, optional benchmark config JSON loaded into `BenchmarkConfig`, and input files `definition.json`, `workload.jsonl`, and `solution.json`.
- Important optional runtime variables include `PYTORCH_ALLOC_CONF`, `PYTORCH_ROCM_ARCH`, `SOLEXECBENCH_ENV_SNAPSHOT`, `SOLEXECBENCH_ENV_SNAPSHOT_PATH`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, `SOL_EXECBENCH_CLOCKS_LOCKED`, `SOL_EXECBENCH_SCLK_LEVEL`, `SOL_EXECBENCH_MCLK_LEVEL`, and `FLASHINFER_TRACE_DIR`; these are read or set in `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/clock_lock.py`, `docker/entrypoint.sh`, and `scripts/run_docker.sh`.
- Docker/dependency diagnostics use `SOL_EXECBENCH_*` override variables in `scripts/run_docker.sh`, `src/sol_execbench/core/dependency_matrix.py`, and `src/sol_execbench/core/runtime_evidence.py`.

**Build:**
- `pyproject.toml` defines package metadata, dependency groups, console scripts, pytest markers, Ruff rules, Ty source scope, uv indexes, and uv source mappings.
- `uv.lock` pins the resolved dependency graph.
- `docker/Dockerfile` defines the ROCm container image, system packages, uv install, virtual environment, ROCm environment variables, and default PyTorch/Triton wheel replacement flow.
- `docker/rocm-targets.json` declares ROCm 7.0.2, 7.1.1, and 7.2.0 container targets with PyTorch ROCm wheel policies.

## Platform Requirements

**Development:**
- Python 3.12 or 3.13 with uv.
- ROCm-capable AMD GPU is needed for live GPU evaluation and tests marked `requires_rocm`, `requires_rdna4`, or `requires_cdna3`.
- Docker-based GPU evaluation requires Docker, ROCm drivers, `/dev/kfd`, `/dev/dri`, `video` group access, and a compatible AMD GPU; `scripts/run_docker.sh` enforces/preflights these requirements.
- Native HIP/C++ solution tests require ROCm compiler tooling, `hipcc`, PyTorch ROCm, and supported ROCm development headers/libraries.

**Production:**
- Not a hosted service. The production-equivalent runtime is a local or containerized benchmark execution environment using `sol-execbench` and optional Docker targets from `docker/rocm-targets.json`.

---

*Stack analysis: 2026-06-01*
