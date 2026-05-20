---
last_mapped: 2026-05-20
last_mapped_commit: unknown
focus: tech
---

# Stack

## Runtime

SOL ExecBench is a Python 3.12+ package named `sol-execbench`, configured in
`pyproject.toml`. The package uses `uv` for dependency management and exposes the
CLI command `sol-execbench` via `sol_execbench.cli:cli`.

## Primary Technologies

- Python package source lives in `src/sol_execbench/`.
- CLI layer uses Click and Rich in `src/sol_execbench/cli/main.py`.
- Data schemas use Pydantic v2 models in `src/sol_execbench/core/data/`.
- GPU evaluation uses PyTorch, CUDA, Triton, CUTLASS, cuDNN, CuTe DSL, cuTile,
  CUPTI, and safetensors.
- Native C++/CUDA builds go through `torch.utils.cpp_extension` from
  `src/sol_execbench/driver/templates/build_ext.py`.
- Timing uses CUPTI activity tracing in `src/sol_execbench/core/bench/timing.py`.

## Dependency Configuration

Runtime dependencies are declared in `pyproject.toml`, including `torch>=2.10.0`,
`torchvision>=0.24.1`, `ninja`, `cuda-tile==1.1.0`,
`nvidia-cudnn-frontend==1.18.0`, `nvidia-cutlass-dsl[cu13]==4.4.1`,
`cupti-python>=13.0.1`, `datasets`, `safetensors`, `click`, and `rich`.

Development dependencies are in the `dev` group: `pytest>=9.0.2` and
`pytest-xdist>=3.5`.

## Packaging

The build backend is Hatchling. `uv.lock` pins the resolved environment.
PyTorch wheels are sourced from the explicit `pytorch-cu130` index in
`pyproject.toml`.

## Container

`docker/Dockerfile` builds on `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04`,
installs CUTLASS v4.4.1, copies `uv`, syncs dependencies into `/venv`, and sets
`PYTHONPATH=/sol-execbench/src` so mounted source overrides the installed package.

## Local Commands

- `uv sync --all-groups` installs dependencies.
- `uv run sol-execbench <problem_dir> --solution solution.json` runs one problem.
- `uv run pytest tests/` runs tests.
- `uv run ruff check .` lints.
- `uv run ruff format .` formats.
- `./scripts/run_docker.sh --build` builds and enters the GPU container.
