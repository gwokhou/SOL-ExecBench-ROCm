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

3. Run the test suite or a focused test before editing behavior.

```bash
uv run pytest tests/
```

4. Use the Docker environment for ROCm GPU evaluation when host setup is not
   already configured.

```bash
./scripts/run_docker.sh --build
```

## Build And Development Commands

| Command | Description |
| --- | --- |
| `uv sync --all-groups` | Install runtime and development dependencies. |
| `uv run sol-execbench <problem_dir> --solution solution.json` | Run the benchmark CLI against one problem. |
| `uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5` | Run a small dataset batch. |
| `uv run pytest tests/` | Run the full adapted pytest suite. |
| `uv run pytest tests/sol_execbench/test_e2e.py` | Run one focused test file. |
| `uv run ruff check .` | Run lint checks. |
| `uv run ruff format .` | Format Python files. |
| `./scripts/run_docker.sh --build` | Build and enter the ROCm Docker environment. |

## Code Style

Ruff is the configured linting and formatting tool. The configuration lives in
`pyproject.toml` under `[tool.ruff]` and `[tool.ruff.lint]`.

- Use Python `>=3.12,<3.14`.
- Use `snake_case` for modules, functions, and variables.
- Use `PascalCase` for classes, Pydantic models, and enum classes.
- Keep changes consistent with nearby modules and avoid broad refactors in
  focused fixes.
- Generated data, downloaded datasets, and examples are excluded from Ruff.

Run style checks with:

```bash
uv run ruff check .
uv run ruff format .
```

## Branch Conventions

No branch naming convention is defined in repository configuration. The
contribution guide requires that changes start from an approved issue and that
commit titles use this format:

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
- Keep pull requests focused on one concern.
- Include tests and documentation for new components.
- Run relevant pytest and Ruff checks before requesting review.
- Sign commits with `git commit -s`.
- Document any ROCm hardware assumptions in tests or PR notes.

## Source Areas

| Area | Purpose |
| --- | --- |
| `src/sol_execbench/cli/` | Click command entry points and terminal reporting. |
| `src/sol_execbench/core/data/` | Public schemas for definitions, workloads, solutions, traces, shapes, and dtypes. |
| `src/sol_execbench/core/bench/` | Correctness, timing, clock locking, IO, and benchmark utilities. |
| `src/sol_execbench/driver/` | Problem staging and generated compile/evaluation templates. |
| `tests/sol_execbench/` | Package-level unit, integration, and migration tests. |
| `tests/examples/` | Example consistency coverage. |
| `tests/docker/dependencies/` | Docker dependency readiness checks. |

