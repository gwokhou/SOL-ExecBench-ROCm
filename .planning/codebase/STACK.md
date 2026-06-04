---
generated_at: 2026-06-04
last_mapped_commit: ac6505f6511818160d36bc6935328ff0bd9468a6
focus: tech
scope: full repo
---

# Technology Stack

## Runtime

- Python package targeting `>=3.12,<3.14`, configured in `pyproject.toml`.
- Package source lives under `src/sol_execbench/`.
- CLI entry points:
  - `sol-execbench` -> `sol_execbench.cli:cli`
  - `sol-execbench-baseline` -> `sol_execbench.cli.baseline:cli`
- Runtime and development dependency management uses `uv`, `uv.lock`, and Hatchling as the build backend.

## Primary Technologies

- CLI and terminal rendering: Click and Rich in `src/sol_execbench/cli/`.
- Data contracts and report schemas: Pydantic v2 models under `src/sol_execbench/core/data/` and other core report modules.
- GPU tensor runtime: PyTorch with ROCm wheels on Linux x86_64.
- Triton GPU programming: `triton-rocm==3.6.0` on Linux x86_64.
- Native ROCm build path: HIP/C++ sources staged by `src/sol_execbench/driver/problem_packager.py` and built through `torch.utils.cpp_extension` via `src/sol_execbench/driver/templates/build_ext.py`.
- Native solution categories: HIP/C++, hipBLAS, MIOpen, Composable Kernel, and rocWMMA.
- Dataset and tensor asset utilities: Hugging Face `datasets`, `safetensors`, and local dataset modules under `src/sol_execbench/core/dataset/`.
- Scoring and evidence utilities: AMD score, AMD bound estimates, SOLAR derivation helpers, runtime evidence, consistency, trust summary, matrix diff, and claim-upgrade helpers under `src/sol_execbench/core/`.

## Dependencies

- Runtime dependencies in `pyproject.toml`:
  - `torch==2.10.0+rocm7.1` and `torchvision==0.25.0+rocm7.1` for Linux x86_64.
  - Non-ROCm `torch==2.10.0` and `torchvision==0.25.0` for non-Linux or non-x86_64 development.
  - `triton-rocm==3.6.0` for Linux x86_64.
  - `ninja`, `pydantic`, `safetensors`, `click`, `rich`, `datasets`, `torch-c-dlpack-ext`, and `apache-tvm-ffi`.
- Development dependencies:
  - `pytest`, `pytest-xdist`, `ruff`, `ty`, and `pre-commit`.
- Dependency indexes:
  - PyPI: `https://pypi.org/simple`
  - PyTorch ROCm 7.1 wheels: `https://download.pytorch.org/whl/rocm7.1`
  - PyTorch wheel root for `triton-rocm`: `https://download.pytorch.org/whl/`

## ROCm Baseline

- Project baseline is ROCm `>=7.0`; the default Python dependency stack is centered on ROCm 7.1 wheels.
- Docker default target is `rocm/dev-ubuntu-24.04:7.1.1-complete`.
- Docker target matrix in `docker/rocm-targets.json` covers:
  - ROCm 7.0.2 with PyTorch `2.10.0+rocm7.0`
  - ROCm 7.1.1 with PyTorch `2.10.0+rocm7.1`
  - ROCm 7.2.0 with PyTorch `2.11.0+rocm7.2`
- ROCm hardware scope targets RDNA 4 and CDNA 3. Supported schema hardware values include `gfx1200`, `gfx940`, `gfx941`, `gfx942`, and `LOCAL`.
- Device access expects `/dev/kfd` and `/dev/dri`; visibility environment variables include `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION`.

## Configuration

- Python packaging, dependency indexes, pytest markers, Ruff, and Ty configuration live in `pyproject.toml`.
- Docker build/runtime configuration lives in `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`, and `scripts/run_docker.sh`.
- GitHub Actions CI is `.github/workflows/code-quality.yml`, running Python 3.12 and 3.13, `uv sync --locked --all-groups`, Ruff, Ty, and CPU-safe pytest subsets.
- Provenance and redistribution policy is configured in `provenance.toml` and documented in `docs/provenance.md` and `docs/compliance.md`.
- Benchmark runtime configuration is represented by `BenchmarkConfig` and device config models under `src/sol_execbench/core/bench/config/`.
- Generated data, downloaded datasets, build outputs, local artifacts, and caches are excluded from normal source/lint scope.

## Repository Areas

- `src/sol_execbench/cli/`: Click command surface for benchmark evaluation, baseline comparison, diagnostics, toolchain reports, and dataset migration.
- `src/sol_execbench/core/data/`: public data schemas for definitions, workloads, solutions, traces, dtypes, shapes, and evaluator contracts.
- `src/sol_execbench/core/bench/`: correctness, timing, runtime, I/O, clock locking, reward-hack detection, `rocprofv3`, and static-kernel evidence helpers.
- `src/sol_execbench/core/dataset/`: dataset categories, manifests, checksums, migration, readiness, run state, sharding, execution closure, denominator, and parity-gap logic.
- `src/sol_execbench/core/scoring/`: AMD-native score, AMD bound estimates, SOLAR derivation, baseline artifacts, and hardware model helpers.
- `src/sol_execbench/driver/`: staging and driver templates used to execute submitted solutions in a subprocess.
- `scripts/`: dataset download/inspection, dataset batch running, compatibility/report generation, prerelease bundle, readiness, and release-candidate validation tools.
- `examples/`: category-specific examples for PyTorch, Triton, HIP/C++, hipBLAS, MIOpen, CK, rocWMMA, and legacy CUDA/NVIDIA compatibility fixtures.
- `tests/`: pytest coverage for schemas, CLI behavior, driver behavior, ROCm checks, examples, Docker dependency checks, evidence reports, scoring, and dataset workflows.
- `docs/`: user, developer, ROCm, schema, timing, provenance, prerelease, release, and research-preview documentation.

## Development Commands

- Install dependencies: `uv sync --all-groups`.
- Run one benchmark: `uv run sol-execbench <problem_dir> --solution <solution-path>`.
- Run a dataset batch: `uv run scripts/run_dataset.py <downloaded-benchmark-dir> --limit 5`.
- Run all tests: `uv run pytest tests/`.
- Run one test file: `uv run pytest tests/sol_execbench/test_e2e.py`.
- Lint: `uv run --with ruff ruff check .`.
- Format: `uv run --with ruff ruff format .`.
- Type check: `uv run ty check`.
- Build/enter ROCm Docker environment: `./scripts/run_docker.sh --build`.
