# Technology Stack

**Analysis Date:** 2026-06-04

## Languages

**Primary:**
- Python 3.12+ - Package implementation in `src/sol_execbench/`, scripts in `scripts/`, and tests in `tests/`; `pyproject.toml` requires `>=3.12,<3.14`, and `.python-version` pins local development to `3.12`.

**Secondary:**
- HIP/C++ - Native benchmark solutions and ROCm library examples under `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, and `examples/rocwmma/`; staged and compiled by `src/sol_execbench/driver/problem_packager.py` and `src/sol_execbench/driver/templates/build_ext.py`.
- Bash - Docker and dataset helper wrappers in `scripts/run_docker.sh`, `scripts/download_data.sh`, and `docker/entrypoint.sh`.
- JSON / JSONL - Public benchmark schemas, trace output, manifests, sidecars, and Docker target definitions in `src/sol_execbench/core/data/`, `docker/rocm-targets.json`, and generated `data/` layouts.
- Markdown - User and release documentation in `README.md` and `docs/`.

## Runtime

**Environment:**
- Python runtime: CPython 3.12-compatible package configured by `pyproject.toml` and `.python-version`.
- GPU runtime: AMD ROCm user space with HIP device access. Native GPU evaluation expects ROCm-capable AMD hardware, `/dev/kfd`, `/dev/dri`, and PyTorch ROCm compatibility APIs as documented in `README.md`.
- Docker runtime: ROCm Ubuntu container workflow based on `rocm/dev-ubuntu-24.04` in `docker/Dockerfile` and target selection from `docker/rocm-targets.json`.

**Package Manager:**
- uv - Project dependency and virtual environment manager used by `pyproject.toml`, `uv.lock`, `docker/Dockerfile`, `scripts/run_docker.sh`, and documented commands in `README.md`.
- Lockfile: present at `uv.lock`; Docker installs use `uv sync --frozen` in `docker/Dockerfile`.

## Frameworks

**Core:**
- Click `>=8.0` - CLI command framework for `sol-execbench` and `sol-execbench-baseline` in `src/sol_execbench/cli/main.py` and `src/sol_execbench/cli/baseline.py`.
- Rich `>=13.0` - CLI tables and progress rendering in `src/sol_execbench/cli/main.py`.
- Pydantic `>=2.12.5` - Strict data models and schema validation in `src/sol_execbench/core/data/`, `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, and compatibility/evidence models.
- PyTorch `2.10.0+rocm7.1` on Linux x86_64 - Tensor execution, HIP-backed timing via `torch.cuda`, and native extension builds in `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/driver/templates/eval_driver.py`, and `src/sol_execbench/driver/templates/build_ext.py`.
- Triton ROCm `3.6.0` on Linux x86_64 - Supported solution language for Triton ROCm examples under `examples/triton/` and schema values in `src/sol_execbench/core/data/solution.py`.

**Testing:**
- pytest `>=9.0.2` - Test runner configured in `pyproject.toml`; tests live under `tests/`.
- pytest-xdist `>=3.5` - Parallel test execution through `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.

**Build/Dev:**
- Hatchling - PEP 517 build backend declared in `pyproject.toml`.
- Ruff `>=0.4` - Linting and formatting configured in `pyproject.toml` and `.pre-commit-config.yaml`.
- ty `>=0.0.39` - Type checking configured by `[tool.ty.src]` in `pyproject.toml` and pre-push hook in `.pre-commit-config.yaml`.
- pre-commit `>=4.6.0` - Local quality hooks in `.pre-commit-config.yaml`.
- Ninja `>=1.13.0` - Native extension build acceleration required by `pyproject.toml` and used through PyTorch extension builds.

## Key Dependencies

**Critical:**
- `torch==2.10.0+rocm7.1` on Linux x86_64 - Required for ROCm tensor execution, HIP-backed `torch.cuda.Event` timing, device probes, and `torch.utils.cpp_extension` builds in `src/sol_execbench/core/bench/timing.py` and `src/sol_execbench/driver/templates/build_ext.py`.
- `torchvision==0.25.0+rocm7.1` on Linux x86_64 - Locked companion package for PyTorch ROCm in `pyproject.toml` and Docker dependency policy in `docker/rocm-targets.json`.
- `triton-rocm==3.6.0` on Linux x86_64 - ROCm Triton kernel support and Docker target policy in `pyproject.toml` and `docker/rocm-targets.json`.
- `pydantic>=2.12.5` - Enforces public schemas, compile-option validation, Matrix models, and evidence sidecars in `src/sol_execbench/core/data/` and `src/sol_execbench/core/compatibility.py`.
- `datasets>=4.8.2` - Hugging Face dataset ingestion for `nvidia/SOL-ExecBench` in `scripts/download_solexecbench.py`.

**Infrastructure:**
- `safetensors>=0.7.0` - Runtime dependency declared in `pyproject.toml` for tensor artifact compatibility.
- `torch-c-dlpack-ext>=0.1.5` - Runtime dependency declared in `pyproject.toml` for DLPack extension support.
- `apache-tvm-ffi>=0.1.9` - Runtime dependency declared in `pyproject.toml`.
- `click>=8.0` and `rich>=13.0` - CLI UX dependencies for command handling and terminal output in `src/sol_execbench/cli/`.
- ROCm command-line tools - `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `rocprofv3`, `amd-smi`, and `rocm-smi` are probed or invoked by `docker/Dockerfile`, `docker/entrypoint.sh`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and `src/sol_execbench/core/bench/clock_lock.py`.

## Configuration

**Environment:**
- CLI evaluation sets `PYTORCH_ALLOC_CONF=expandable_segments:True` for compile/eval subprocesses in `src/sol_execbench/cli/main.py`.
- Native PyTorch extension builds set `PYTORCH_ROCM_ARCH` from solution `target_hardware` when unset in `src/sol_execbench/driver/templates/build_ext.py`.
- Optional environment snapshots are controlled by `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` in `src/sol_execbench/cli/main.py`.
- GPU visibility variables `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION` are recorded by `src/sol_execbench/core/environment.py` and Docker runtime evidence in `scripts/run_docker.sh`.
- Clock and Docker diagnostics use `SOL_EXECBENCH_*` variables documented in `docs/CONFIGURATION.md` and implemented by `scripts/run_docker.sh`, `docker/entrypoint.sh`, and `src/sol_execbench/core/bench/clock_lock.py`.

**Build:**
- `pyproject.toml` defines package metadata, dependencies, uv indexes, pytest markers, Ruff rules, ty source roots, and console scripts.
- `uv.lock` pins resolved dependencies.
- `docker/Dockerfile` defines the ROCm container build, uv environment, PyTorch ROCm wheel installation, and runtime `PYTHONPATH`.
- `docker/rocm-targets.json` declares supported ROCm container targets and PyTorch/Triton ROCm wheel policies.
- `.pre-commit-config.yaml` defines Ruff, Ruff format, DCO sign-off, and ty hooks.

## Platform Requirements

**Development:**
- Python 3.12.
- uv with `uv sync --all-groups` for local development dependencies.
- Linux x86_64 resolves ROCm PyTorch/Triton wheels from `https://download.pytorch.org/whl/rocm7.1`; non-Linux and non-x86_64 environments resolve non-ROCm PyTorch wheels for CPU-safe development tasks through `pyproject.toml`.
- ROCm-capable AMD hardware is required for real GPU evaluation; GPU-marked tests use pytest markers from `pyproject.toml`.

**Production:**
- Not a deployed web service. The production-like target is a local or containerized ROCm benchmark runner using `sol-execbench`, `scripts/run_dataset.py`, and `scripts/run_docker.sh`.
- Container targets are declared for ROCm 7.0.2, 7.1.1, and 7.2.0 in `docker/rocm-targets.json`; the default target is `rocm-7.1.1-ubuntu-24.04-container`.

---

*Stack analysis: 2026-06-04*
