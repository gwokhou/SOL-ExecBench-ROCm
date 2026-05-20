---
last_mapped: 2026-05-20
last_mapped_commit: unknown
focus: quality
---

# Conventions

## Python Style

The repository uses Ruff for linting and formatting. Configuration is in
`pyproject.toml`, with generated and heavy data paths excluded. Rule `E741` is
ignored. Pre-commit hooks in `.pre-commit-config.yaml` run Ruff and Ruff format.

## Source Headers

Most source files include NVIDIA copyright and Apache-2.0 SPDX headers. New
source files should follow nearby files when adding production code.

## Type And Model Style

Pydantic schema classes live in `src/sol_execbench/core/data/` and generally
inherit from `BaseModelWithDocstrings`. Validators use Pydantic v2
`@model_validator` and `@field_validator`. Dataclass configuration objects live
under `src/sol_execbench/core/bench/config/`.

## Naming

- Modules, functions, and variables use `snake_case`.
- Classes and Pydantic models use `PascalCase`.
- Enum classes use `PascalCase`; enum members use uppercase names with string
  values matching JSON schema values.
- Tests use descriptive `test_*` names and class groupings like
  `TestCheckMonkeyPatch`.

## Error Handling

CLI user errors are raised as `click.ClickException` in
`src/sol_execbench/cli/main.py`. Runtime evaluation failures are represented as
`EvaluationStatus` values in `src/sol_execbench/core/data/trace.py` and emitted
as trace records rather than raw crashes when possible.

## Subprocess Boundaries

The CLI and tests call subprocesses with argument lists, not shell strings.
Staging scripts use files in the current staging directory as their contract.
The eval driver keeps stdout reserved for trace JSON and routes library output
to stderr.

## Security Patterns

Source paths are validated against absolute paths and `..` traversal in
`SourceFile`. Python submissions are prevented from using inline C++ extension
loading inside the eval subprocess. Reward-hack checks protect timing,
threading, lazy outputs, and eval-driver monkey patching.

## Documentation

Schema behavior is documented in `docs/definition.md`, `docs/workload.md`,
`docs/solution.md`, and `docs/trace.md`. Contributor guidance is in
`CONTRIBUTING.md` and agent guidance is in `AGENTS.md`.
