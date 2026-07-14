# Repository Guidelines

## Project Structure & Module Organization

The package source is under `src/sol_execbench/`: `cli/` contains Click commands,
`core/` contains models and benchmark logic, and `driver/` contains execution
backends. Put package tests in `tests/sol_execbench/`, CLI tests beside their
commands, and example workflow tests in `tests/examples/`. Runnable kernel examples
live in `examples/`; Docker support is in `docker/`; helper scripts are in
`scripts/`. Keep downloaded benchmark data and generated outputs in `data/`, not in
commits.

## Reuse and Shared-Primitives Policy

- Before adding a generic helper, search with `rg` and reuse the existing
  implementation; do not hand-write duplicates in domains, CLIs, or scripts.
- Shared helpers belong in `core/` by concern: `integrity`, `process`,
  `data.json_utils`, `text_utils`, `platform.runtime`, `arguments`, or
  `timestamps`. Do not create `core.utils` or restore the retired evidence
  checksum/log modules.
- `tools/` is only for external-tool integrations. Domain packages must not
  re-export generic helpers; wrappers require a documented test seam and a
  focused test. Migrate in-scope duplicates and confirm old imports are gone.

## Build, Test, and Development Commands

- `uv sync --all-groups` installs runtime and development dependencies.
- `uv run pytest tests/` runs the test suite. Do not add `-n auto`: ROCm workers
  consume substantial memory; the repository config limits the default to eight.
- `uv run pytest tests/sol_execbench/test_e2e.py` runs a focused test module.
- `uv run sol-execbench <problem_dir> --solution <path>` evaluates one solution.
- `uv run --with ruff ruff check .` lints, and `uv run --with ruff ruff format .`
  formats Python files. Run `uv run ty` for static type checking when relevant.
- `./scripts/run_docker.sh --build` builds the ROCm evaluation environment.

## Coding Style & Naming Conventions

Target Python 3.12+ and follow Ruff's formatting. Use four-space indentation,
`snake_case` for modules, functions, and variables, and `PascalCase` for classes
and Pydantic models. Keep focused changes local to the affected subsystem; avoid
unrelated refactors. Name tests descriptively, for example
`test_rejects_invalid_solution_schema`.

In production code, do not embed large HIP or C++ source files directly in
Python string literals. Store them as package resources and load them through
`importlib.resources`. Tests may embed source snippets or fixtures directly when
that keeps the test focused and readable.

## Testing Guidelines

Use Pytest and place coverage near the implementation. Mark environment-specific
tests with existing markers such as `requires_rocm`, `cpp`, `requires_rdna4`, or
`requires_cdna3`; these document required hardware and allow safe skipping. Prefer
small unit tests for schemas and driver logic, adding integration coverage for
subprocess or GPU execution changes.

## Commit & Pull Request Guidelines

Use concise imperative commit summaries and sign commits with DCO, for example:
`git commit -s -m "Fix trace parsing"`. Keep PRs focused; describe behavior
changes and list tests and any ROCm hardware checks performed.

## Security & Configuration

Never commit tokens, proprietary kernels, datasets, caches, or benchmark output.
GPU evaluation requires ROCm-capable AMD hardware and may need `/dev/kfd` and
`/dev/dri` access; document architecture-specific assumptions in tests or PR notes.

## Container Permission Handling

When a required container operation fails solely because of sandbox, cache, or
filesystem permissions, request the necessary escalation from the user and retry
the same scoped operation. Do not treat that permission failure as a benchmark,
calibration, or implementation failure.

For any GPU-related operation (enumeration, compilation, execution, profiling,
tracing, or calibration), a sandbox may expose `/dev/kfd` incompletely or hide
`/dev/dri/renderD*` and other required runtime resources. Before concluding that
a GPU, architecture, counter, driver, compiler, or runtime feature is
unsupported or broken, request narrowly scoped host execution for the exact
failing command and retry it there. Keep the command bounded (single workload or
explicit limit), write generated evidence only to an approved temporary/output
directory, and do not use service-manager or other sandbox-escape wrappers as a
substitute for that approval.
