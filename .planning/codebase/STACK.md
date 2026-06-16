---
generated_by: gsd-map-codebase
focus: tech
mapped_at: 2026-06-16
---

# Technology Stack

## Runtime

- Python package using Python `>=3.12,<3.14`, configured in `pyproject.toml`.
- Package source lives under `src/sol_execbench/`.
- Build backend is Hatchling through `[build-system]` in `pyproject.toml`.
- Dependency and command execution are managed with `uv`.
- Public console scripts:
  - `sol-execbench = sol_execbench.cli:cli`
  - `sol-execbench-baseline = sol_execbench.cli.baseline:cli`

## Primary Libraries

- `torch` and `torchvision` are pinned to ROCm wheels on Linux x86_64:
  `torch==2.10.0+rocm7.1` and `torchvision==0.25.0+rocm7.1`.
- Non-Linux or non-x86_64 environments resolve non-ROCm `torch==2.10.0` and
  `torchvision==0.25.0` wheels for CPU-safe development.
- `triton-rocm==3.6.0` is enabled on Linux x86_64 through the PyTorch wheel
  index in `pyproject.toml`.
- `pydantic>=2.12.5` models public schemas in `src/sol_execbench/core/data/`.
- `click>=8.0` and `rich>=13.0` implement CLI parsing and terminal output in
  `src/sol_execbench/cli/main.py`.
- `safetensors>=0.7.0` supports workload inputs in
  `src/sol_execbench/core/bench/io.py`.
- `datasets>=4.8.2` supports dataset workflows.
- `ninja>=1.13.0`, `torch-c-dlpack-ext>=0.1.5`, and
  `apache-tvm-ffi>=0.1.9` are runtime dependencies.

## Development Tooling

- Dev dependency group in `pyproject.toml` includes `pytest`, `pytest-xdist`,
  `ruff`, `ty`, and `pre-commit`.
- Ruff lint/format configuration lives in `[tool.ruff]` and `[tool.ruff.lint]`.
- Ty source scope is `src` and `tests` via `[tool.ty.src]`.
- Pytest default options are `-n 8 --dist loadgroup` to avoid PyTorch/ROCm
  worker memory blowups.
- GitHub Actions workflow `.github/workflows/code-quality.yml` runs on Python
  3.12 and 3.13 with `uv sync --locked --all-groups`, Ruff, Ty, CPU-safe package
  tests, and example consistency tests.
- `.pre-commit-config.yaml` runs Ruff check/format, a DCO sign-off check, and a
  pre-push Ty check.

## GPU And Native Tooling

- ROCm is the only supported GPU runtime target.
- Native solution categories include HIP/C++, hipBLAS, MIOpen, Composable
  Kernel, and rocWMMA through `src/sol_execbench/core/data/solution.py`.
- Native extension builds are staged by `src/sol_execbench/driver/problem_packager.py`
  and `src/sol_execbench/driver/templates/build_ext.py`.
- Evaluation runtime uses HIP-backed PyTorch through the historical
  `torch.cuda` namespace.
- `rocprofv3` profiling integration lives in
  `src/sol_execbench/core/bench/rocm_profiler.py`.
- Static kernel artifact extraction lives in
  `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- Clock locking and timing policy helpers live in
  `src/sol_execbench/core/bench/clock_lock.py`,
  `src/sol_execbench/core/bench/timing.py`, and
  `src/sol_execbench/core/bench/timing_policy.py`.

## Configuration Sources

- Project metadata, dependencies, pytest markers, Ruff, Ty, and uv indexes:
  `pyproject.toml`.
- Docker target metadata: `docker/rocm-targets.json`.
- Docker image and runtime setup: `docker/Dockerfile`, `docker/entrypoint.sh`,
  and `scripts/run_docker.sh`.
- Benchmark runtime config model: `src/sol_execbench/core/bench/config/`.
- Packaged AMD hardware model data: `src/sol_execbench/data/amd_hardware_models/gfx1200.json`.
- Local downloaded datasets and benchmark assets belong under `data/`.

## Common Commands

- Install dependencies: `uv sync --all-groups`.
- Run one problem: `uv run sol-execbench <problem_dir> --solution <solution-path>`.
- Compare traces: `uv run sol-execbench-baseline --candidate <file> --baseline <file>`.
- Run dataset batch: `uv run scripts/run_dataset.py <downloaded-benchmark-dir> --limit 5`.
- Run full tests: `uv run pytest tests/`.
- Lint: `uv run --with ruff ruff check .`.
- Format: `uv run --with ruff ruff format .`.
- Build/run ROCm container: `./scripts/run_docker.sh --build`.
