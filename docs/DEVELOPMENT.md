<!-- generated-by: gsd-doc-writer -->
# Development

This project is a Python 3.12+ package managed with `uv`. Development work
should preserve the ROCm-only scope and the public benchmark schemas unless a
ROCm-specific change is unavoidable.

## Local Setup

1. Fork or clone the repository.

```bash
git clone <repository-url>
cd SOL-ExecBench-ROCm
```

2. Install dependencies.

```bash
uv sync --all-groups
```

3. Run the test suite or a focused test before editing behavior. Pytest is
   configured to use `pytest-xdist` with `-n auto --dist loadgroup`.

```bash
uv run pytest tests/
```

4. Optional: install the repository pre-commit hooks if you want Ruff fixes,
   Ruff formatting, and the DCO sign-off check to run before commits.

```bash
uv run --with pre-commit pre-commit install
uv run --with pre-commit pre-commit install --hook-type commit-msg
```

5. Use the Docker environment for ROCm GPU evaluation when host setup is not
   already configured.

```bash
./scripts/run_docker.sh --build
```

## Build And Development Commands

| Command | Description |
| --- | --- |
| `uv sync --all-groups` | Install runtime and development dependencies. |
| `uv run sol-execbench <problem_dir> --solution <solution-path>` | Run the benchmark CLI against one problem. |
| `uv run sol-execbench-baseline <problem_dir>` | Run the baseline CLI entry point. |
| `uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5` | Run a small dataset batch. |
| `uv run pytest tests/` | Run the full adapted pytest suite. |
| `uv run pytest tests/sol_execbench/test_e2e.py` | Run one focused test file. |
| `uv run ruff check .` | Run lint checks. |
| `uv run ruff format .` | Format Python files. |
| `uv run ty check` | Run type checks over `src` and `tests`. |
| `./scripts/run_docker.sh --build` | Build and enter the ROCm Docker environment. |

There is no dedicated package build script in `pyproject.toml`; the package uses
Hatchling as its build backend.

## Code Style

Ruff is the configured linting and formatting tool. The configuration lives in
`pyproject.toml` under `[tool.ruff]` and `[tool.ruff.lint]`.
Ty is the configured type checker. Its source roots live under `[tool.ty.src]`.
Pre-commit hooks in `.pre-commit-config.yaml` run Ruff lint fixes, Ruff
formatting, and a local DCO sign-off check for commit messages.

- Use Python `>=3.12,<3.14`.
- Use `snake_case` for modules, functions, and variables.
- Use `PascalCase` for classes, Pydantic models, and enum classes.
- Keep changes consistent with nearby modules and avoid broad refactors in
  focused fixes.
- Generated data, downloaded datasets, and examples are excluded from Ruff.
- PyTorch and torchvision resolve from the ROCm 7.1 wheel index on Linux and
  Windows through `tool.uv.sources`.

Run style checks with:

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
```

The `Python Quality` GitHub Actions workflow runs `uv sync --locked --all-groups`,
Ruff, Ty, and CPU-safe pytest checks for Python 3.12 and 3.13 on every push and
pull request.

## Branch Conventions

The current checkout is on `main`. No branch naming convention is defined in
repository configuration, and there is no `.github/PULL_REQUEST_TEMPLATE.md`
file in this checkout. The contribution guide requires that changes start from
an approved issue and that commit titles use this format:

```text
#<Issue Number> - <Commit Title>
```

Commits should be signed with DCO sign-off:

```bash
git commit -s -m "#123 - Fix trace parsing"
```

## PR Process

The root `CONTRIBUTING.md` describes the contribution process. In practice:

- Start from an approved GitHub issue before opening a code contribution.
- Fork the repository, push changes to a branch on the fork, and open a pull
  request from that branch.
- Keep pull requests focused on one concern.
- Include tests and documentation for new components.
- Run relevant pytest, Ruff, and Ty checks before requesting review.
- Expect the `Python Quality` workflow to run Ruff, Ty, and CPU-safe pytest
  checks on the pull request.
- Sign commits with `git commit -s`.
- Ensure local build and test logs are clean before submitting.
- Document any ROCm hardware assumptions in tests or PR notes.

## Test Markers For Development

The pytest marker definitions are split between `pyproject.toml` and
`tests/conftest.py`, with additional runtime skip behavior in `tests/conftest.py`.

| Marker | Purpose |
| --- | --- |
| `cpp` | Marks tests that compile HIP/C++ extensions and require the ROCm compiler toolchain. |
| `timing_serial` | Marks GPU timing tests that are skipped by default; run them with `pytest tests -m timing_serial -n 0`. |
| `requires_rocm` | Marks tests that require a ROCm GPU visible through PyTorch. |
| `requires_rocm_dev` | Marks tests that require ROCm native extension development headers under `/opt/rocm`. |
| `requires_ck` | Marks tests that require Composable Kernel headers. |
| `requires_rocwmma` | Marks tests that require rocWMMA headers. |
| `requires_rdna4` | Marks tests that require an AMD RDNA 4 GPU such as `gfx1200`. |
| `requires_cdna3` | Marks tests that require an AMD CDNA 3 GPU such as `gfx942`. |
| `requires_cutile` | Legacy NVIDIA cuTile marker that is skipped in this ROCm-only port. |

## Source Areas

| Area | Purpose |
| --- | --- |
| `src/sol_execbench/cli/` | Click command entry points and terminal reporting. |
| `src/sol_execbench/core/data/` | Public schemas for definitions, workloads, solutions, traces, shapes, and dtypes. |
| `src/sol_execbench/core/bench/` | Correctness, timing, clock locking, IO, and benchmark utilities. |
| `src/sol_execbench/core/dataset/` | Dataset discovery and metadata helpers. |
| `src/sol_execbench/core/scoring/` | AMD scoring, bound estimates, and derived evidence helpers. |
| `src/sol_execbench/driver/` | Problem staging and generated compile/evaluation templates. |
| `src/sol_execbench/data/` | Packaged AMD hardware model data. |
| `tests/sol_execbench/` | Package-level unit, integration, and migration tests. |
| `tests/examples/` | Example consistency coverage. |
| `tests/docker/dependencies/` | Docker dependency readiness checks. |
