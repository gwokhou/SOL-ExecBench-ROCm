<!-- generated-by: gsd-doc-writer -->
# Contributing

Thank you for contributing to SOL ExecBench ROCm Port. This repository is a
ROCm-only Python port of SOL ExecBench, so contributions should preserve the
benchmark schemas and evaluation semantics unless a ROCm-specific change is
required.

## Development Setup

See [GETTING-STARTED.md](docs/GETTING-STARTED.md) for prerequisites and first-run
instructions, and [DEVELOPMENT.md](docs/DEVELOPMENT.md) for local development
setup.

The short local setup path is:

```bash
uv sync --all-groups
uv run pytest tests/
uv run --with ruff ruff check .
```

For ROCm GPU evaluation, use a ROCm-capable AMD system or the repository Docker
helper:

```bash
./scripts/run_docker.sh --build
```

## Coding Standards

- Use Python `>=3.12,<3.14`, as configured in `pyproject.toml`.
- Run Ruff before submitting changes:

  ```bash
  uv run --with ruff ruff check .
  uv run --with ruff ruff format .
  ```

- The Ruff pre-commit hooks are configured in `.pre-commit-config.yaml`; they
  run Ruff lint fixes and formatting.
- Keep changes consistent with nearby modules. Use `snake_case` for modules,
  functions, and variables, and `PascalCase` for classes, Pydantic models, and
  enum classes.
- Do not commit local caches, build artifacts, downloaded benchmark datasets, or
  generated benchmark output.

No GitHub Actions workflow is present in this checkout, so local pytest and Ruff
results are the primary validation record for a pull request.

## PR Guidelines

- Start from an approved GitHub issue before opening a code contribution.
- Keep each pull request focused on one concern.
- Include tests for behavior changes. Place package tests under
  `tests/sol_execbench/` and example workflow tests under `tests/examples/`.
- Include documentation when adding or changing public commands, schemas,
  examples, or ROCm hardware assumptions.
- Run the relevant pytest and Ruff commands before requesting review. For GPU
  behavior, list the ROCm hardware and Docker checks you ran.
- Use imperative commit titles in this format:

  ```text
  #<Issue Number> - <Commit Title>
  ```

- Sign commits with DCO sign-off:

  ```bash
  git commit -s -m "#123 - Fix trace parsing"
  ```

The `.pre-commit-config.yaml` file includes a commit-message hook that checks for
a `Signed-off-by:` line.

## Issue Reporting

Use GitHub Issues for bug reports and feature requests:

```text
https://github.com/gwokhou/SOL-ExecBench-ROCm/issues
```

No `.github/ISSUE_TEMPLATE/` directory is present in this checkout. Include the
following information when reporting an issue:

- A clear summary of the bug, feature request, or compatibility gap.
- Steps to reproduce the behavior, including the command you ran.
- Expected and actual behavior.
- Python, PyTorch, ROCm, GPU architecture, and operating system details.
- Relevant trace files, logs, or benchmark problem paths when available.

Do not include proprietary kernels, credentials, Hugging Face tokens, downloaded
datasets, or other private data in issues or pull requests.
