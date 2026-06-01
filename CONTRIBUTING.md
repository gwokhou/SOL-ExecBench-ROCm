<!-- generated-by: gsd-doc-writer -->
# Contributing

Thank you for contributing to SOL ExecBench ROCm Port. This repository is a
ROCm-only Python port of SOL ExecBench, so changes should preserve public
benchmark schemas and evaluation semantics unless a ROCm-specific difference is
required.

## Start From an Issue

Contributions should start from an approved GitHub issue. Keep each pull
request focused on one concern and link the issue in the PR description.

## Development Setup

Install dependencies:

```bash
uv sync --all-groups
```

Run core checks:

```bash
uv run ruff check .
uv run ty check
uv run pytest tests/
```

For ROCm GPU evaluation, use a ROCm-capable AMD host or the Docker helper:

```bash
./scripts/run_docker.sh --build
```

See [Getting Started](docs/GETTING-STARTED.md), [Development](docs/DEVELOPMENT.md),
and [Testing](docs/TESTING.md) for more detail.

## Coding Standards

- Use Python `>=3.12,<3.14`; `.python-version` pins local development to
  Python 3.12.
- Follow Ruff formatting and linting from `pyproject.toml`.
- Run Ty type checks over `src` and `tests`.
- Use `snake_case` for modules, functions, and variables.
- Use `PascalCase` for classes, Pydantic models, and enum classes.
- Keep changes consistent with nearby code.
- Avoid broad refactors in focused fixes.
- Do not commit local caches, virtual environments, build artifacts, downloaded
  datasets, generated benchmark output, credentials, or proprietary kernels.

Format and lint:

```bash
uv run ruff format .
uv run ruff check .
```

Type check:

```bash
uv run ty check
```

The `Python Quality` GitHub Actions workflow enforces Ruff, Ty, CPU-safe package
tests, and example consistency tests on Python 3.12 and 3.13.

Optional pre-commit setup installs Ruff hooks and the commit-message DCO check:

```bash
uv run --with pre-commit pre-commit install
uv run --with pre-commit pre-commit install --hook-type commit-msg
```

## Tests

Place new package tests under `tests/sol_execbench/` near related coverage.
Place example workflow tests under `tests/examples/`.

Use environment-sensitive markers where appropriate:

- `cpp`
- `timing_serial`
- `requires_rocm`
- `requires_rocm_dev`
- `requires_ck`
- `requires_rocwmma`
- `requires_rdna4`
- `requires_cdna3`
- `requires_cutile` for legacy NVIDIA-only coverage that should be skipped in
  this ROCm-only port

For hardware-sensitive changes, record the ROCm version, GPU architecture,
container target if used, and exact test commands in the PR.

## Documentation

Update documentation when changing:

- Public CLI behavior
- Benchmark schemas
- Solution language categories
- Docker targets or runtime assumptions
- ROCm hardware support claims
- Scoring, timing, profiling, or evidence semantics
- Dataset reuse, execution-closure, failure-mode, or sharding semantics
- User-facing examples

Use conservative wording for unvalidated hardware or infrastructure claims.

## Commit Messages

Use imperative commit titles in this project format:

```text
#<Issue Number> - <Commit Title>
```

Sign commits with DCO sign-off:

```bash
git commit -s -m "#123 - Fix trace parsing"
```

The repository pre-commit configuration includes a commit-message hook that
checks for a `Signed-off-by:` line.

## PR Guidelines

PRs should include:

- Linked approved issue
- Clear summary of behavior changes
- Tests, lint, type checks, Docker checks, or GPU checks run
- Documentation updates for public behavior changes
- Hardware assumptions for ROCm-sensitive changes
- Notes about any remaining validation gap

No repository PR template is currently present under `.github/`, so include
these details directly in the pull request description.

The `Python Quality` GitHub Actions workflow runs `uv sync --locked
--all-groups`, Ruff, Ty, CPU-safe package tests, and example consistency tests
on Python 3.12 and 3.13.

## Issue Reporting

When reporting a bug or compatibility gap, include:

- Command run
- Expected behavior
- Actual behavior
- Python version
- PyTorch and ROCm versions
- GPU architecture, such as `gfx1200` or `gfx942`
- Operating system and container target, if relevant
- Trace JSONL, sidecars, logs, or sample problem paths when safe to share

Do not include credentials, Hugging Face tokens, proprietary kernels, private
datasets, or other sensitive data.

No repository issue templates are currently present under `.github/`, so include
the relevant environment and reproduction details in the issue body.
