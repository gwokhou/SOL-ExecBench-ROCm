---
generated_at: 2026-06-02
last_mapped_commit: 8019adc6295a78d4636037889245abcb3f9a52bb
focus: tech
---

# Technology Stack

## Runtime

- Python package targeting `>=3.12,<3.14`, configured in `pyproject.toml`.
- Package source lives under `src/sol_execbench/`.
- Console scripts:
  - `sol-execbench` -> `sol_execbench.cli:cli`
  - `sol-execbench-baseline` -> `sol_execbench.cli.baseline:cli`
- Runtime dependency management uses `uv` and `uv.lock`.

## Primary Libraries

- CLI and terminal UI: Click and Rich in `src/sol_execbench/cli/main.py`.
- Data validation and schemas: Pydantic v2 models in `src/sol_execbench/core/data/`.
- GPU tensor runtime: PyTorch ROCm, with Linux/Windows wheels pinned through the `pytorch-rocm71` uv index.
- Triton ROCm: `triton-rocm==3.6.0` on Linux.
- Native build path: `torch.utils.cpp_extension` through driver packaging and build templates in `src/sol_execbench/driver/`.
- Native ROCm libraries and categories: HIP/C++, hipBLAS, MIOpen, Composable Kernel, and rocWMMA.
- Dataset utilities: Hugging Face `datasets`, `safetensors`, and project-local dataset helpers under `src/sol_execbench/core/dataset/`.

## ROCm Baseline

- Project baseline is ROCm `>=7.0`; current dependency lock is centered on PyTorch `2.10.0+rocm7.1`.
- Docker default target is ROCm `7.1.1-complete` in `docker/Dockerfile`.
- Docker target matrix lives in `docker/rocm-targets.json` and includes 7.0.2, 7.1.1, and 7.2.0 user-space targets.
- Device access is expected through `/dev/kfd` and `/dev/dri`, with GPU markers in `tests/conftest.py`.

## Configuration

- Python tooling is configured in `pyproject.toml`.
- CI is defined in `.github/workflows/code-quality.yml`.
- Docker launch and compatibility preflight logic is in `scripts/run_docker.sh`.
- Release and evidence scripts live in `scripts/`, including `scripts/build_prerelease_artifact_bundle.py`, `scripts/check_prerelease_readiness.py`, and `scripts/release_candidate_validation.py`.
- Provenance policy and source-header classes are configured in `provenance.toml`.

## Static Assets

- AMD hardware model data is under `src/sol_execbench/data/amd_hardware_models/`.
- Examples are grouped by implementation category under `examples/`.
- Downloaded benchmark assets belong under `data/` and should not be committed.
- Generated or local output belongs under ignored locations such as `out/`, `.artifacts/`, `dist/`, and cache directories.

## Development Commands

- Install dependencies: `uv sync --all-groups`.
- Run CLI: `uv run sol-execbench <problem_dir> --solution <solution-path>`.
- Run small dataset batch: `uv run scripts/run_dataset.py <benchmark-dir> --limit 5`.
- Run tests: `uv run pytest tests/`.
- Lint: `uv run ruff check .`.
- Type check: `uv run ty check`.
- Docker ROCm environment: `./scripts/run_docker.sh --build`.
