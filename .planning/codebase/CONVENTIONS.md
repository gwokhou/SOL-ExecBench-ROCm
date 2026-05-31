# Coding Conventions

**Analysis Date:** 2026-05-31

## Naming Patterns

**Files:**
- Use lowercase `snake_case.py` modules under `src/sol_execbench/`, such as `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/core/data/base_model.py`, and `src/sol_execbench/core/scoring/amd_sol_v2.py`.
- Use `test_*.py` test files under `tests/`, placed near the related source area, such as `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/core/bench/test_clock_lock.py`, and `tests/sol_execbench/driver/test_eval_driver.py`.
- Use package `__init__.py` files for import boundaries in `src/sol_execbench/`, `src/sol_execbench/core/`, `tests/sol_execbench/core/`, and related packages.

**Functions:**
- Use `snake_case` for public functions, private helpers, fixtures, and CLI helpers.
- Prefix internal helpers with `_`, as in `_load_solution` in `src/sol_execbench/cli/main.py`, `_run_checked_rocm_smi` in `src/sol_execbench/core/bench/clock_lock.py`, and `_make_spec` in `tests/sol_execbench/core/data/test_solution.py`.
- Prefer small parsing, formatting, validation, and builder helpers over inline logic when a module has repeated schema/report handling, as in `src/sol_execbench/core/scoring/amd_sol_v2.py`.

**Variables:**
- Use `snake_case` for locals, parameters, fixture names, and instance attributes.
- Use uppercase module constants for schemas, environment variable names, marker-like vocabularies, and static options, such as `ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION` in `src/sol_execbench/core/environment.py`, `ENV_SNAPSHOT_ENABLE_ENV` in `src/sol_execbench/cli/main.py`, and `ROCM_SMI_FAILURE_MARKERS` in `src/sol_execbench/core/bench/clock_lock.py`.
- Use leading underscore constants for module-private test data and implementation internals, such as `_MINIMAL_DEFINITION` in `tests/sol_execbench/driver/test_eval_driver.py`.

**Types:**
- Use `PascalCase` for classes, dataclasses, enum classes, Pydantic models, and typed report objects, such as `EnvironmentSnapshot` in `src/sol_execbench/core/environment.py`, `SupportedLanguages` in `src/sol_execbench/core/data/solution.py`, and `BaseModelWithDocstrings` in `src/sol_execbench/core/data/base_model.py`.
- Use uppercase enum members with lowercase string values for externally serialized schema fields, as in `SupportedLanguages.PYTORCH = "pytorch"` in `src/sol_execbench/core/data/solution.py`.
- Use explicit type aliases when they clarify model contracts, such as `NonEmptyString` in `src/sol_execbench/core/data/base_model.py` and `ProbeRunner` in `src/sol_execbench/core/environment.py`.

## Code Style

**Formatting:**
- Use Ruff formatting. The repository config is in `pyproject.toml` under `[tool.ruff]`, and the pre-commit formatter hook is in `.pre-commit-config.yaml`.
- Keep Python source compatible with `requires-python = ">=3.12,<3.14"` from `pyproject.toml`.
- Keep generated data, downloaded datasets, examples, build outputs, caches, and virtual environments out of Ruff scope; the exclusions live in `pyproject.toml`.
- Preserve SPDX/license headers in modules that already have them, especially source under `src/sol_execbench/` and many tests under `tests/sol_execbench/`.

**Linting:**
- Use Ruff linting through `uv run ruff check .`; `.pre-commit-config.yaml` runs Ruff with `--fix` and then `ruff-format`.
- The only explicit lint ignore currently configured is `E741` in `pyproject.toml`.
- Use Ty for static type checks over `src` and `tests`; `pyproject.toml` configures `[tool.ty.src] include = ["src", "tests"]`.
- CI runs `uv run ruff check .`, `uv run ty check`, CPU-safe package tests, and example consistency tests in `.github/workflows/code-quality.yml`.

## Import Organization

**Order:**
1. Optional `from __future__ import annotations`, used in modules such as `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/environment.py`, and `tests/sol_execbench/driver/test_eval_driver.py`.
2. Standard library imports, grouped alphabetically or by nearby convention, such as `json`, `os`, `subprocess`, `Path`, and `typing` imports.
3. Third-party imports, such as `click`, `pytest`, `torch`, `pydantic`, and `rich`.
4. First-party imports from `sol_execbench` or relative package imports.

**Path Aliases:**
- No custom import path alias is configured in `pyproject.toml`.
- Import package code through `sol_execbench...` from tests, for example `from sol_execbench.core.data.solution import BuildSpec` in `tests/sol_execbench/core/data/test_solution.py`.
- Use relative imports inside package modules when importing sibling package APIs, such as `from .data.base_model import BaseModelWithDocstrings` in `src/sol_execbench/core/environment.py` and `from ..core import Definition` in `src/sol_execbench/cli/main.py`.
- Use shared test helper imports from `tests/sol_execbench_type_helpers.py` for schema-heavy tests.

## Error Handling

**Patterns:**
- Raise `ValueError` for invalid domain data, schema payloads, unsupported modes, and parse failures, as in `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/baseline.py`, and `src/sol_execbench/core/scoring/amd_sol_v2.py`.
- Raise `click.ClickException` for user-facing CLI input errors, as in `_resolve_problem_dir`, `_evaluate_cli`, `_contract_cli`, and `_doctor_cli` in `src/sol_execbench/cli/main.py`.
- Wrap lower-level exceptions with context and `from exc` where the original cause helps debugging, such as JSON and AST parsing in `src/sol_execbench/core/baseline.py` and `src/sol_execbench/core/scoring/amd_bound_graph.py`.
- Treat optional diagnostic evidence as non-fatal. CLI sidecar writers in `src/sol_execbench/cli/main.py` catch broad exceptions, print yellow warnings, and return `None` without changing benchmark correctness.
- Hardware/tool availability probes return structured unavailable/failed statuses or `False` rather than crashing when possible, as in `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/clock_lock.py`, and `tests/conftest.py`.
- Tests assert exact exception classes and key message fragments with `pytest.raises(..., match=...)`, as in `tests/sol_execbench/core/data/test_solution.py`.

## Logging

**Framework:** Python `logging` plus Rich console output in CLI modules.

**Patterns:**
- Use `logging.getLogger(__name__)` for library modules that need runtime diagnostics, such as `src/sol_execbench/core/bench/clock_lock.py`.
- Use `logger.warning` for recoverable runtime/tooling failures and `logger.info` for successful clock-lock operations in `src/sol_execbench/core/bench/clock_lock.py`.
- Use `rich.console.Console(stderr=True)` for CLI progress, warnings, success messages, and tables in `src/sol_execbench/cli/main.py`.
- Keep diagnostic sidecar logging explicit and non-authoritative; messages in `src/sol_execbench/cli/main.py` report saved/skipped environment, profile, and static-evidence metadata without altering score authority.

## Comments

**When to Comment:**
- Preserve module docstrings that define purpose and authority boundaries, such as `src/sol_execbench/core/environment.py` and `tests/sol_execbench/driver/test_eval_driver.py`.
- Use short comments for non-obvious compatibility behavior, ROCm-specific semantic boundaries, and benchmark-authority caveats, such as timing compatibility comments in `src/sol_execbench/core/bench/timing.py`.
- Avoid comments that restate simple assignments. Prefer clear helper names and typed models for ordinary logic.

**JSDoc/TSDoc:**
- Not applicable; this is a Python project.
- Use Python docstrings for public classes, models, functions, pytest fixtures, and complex helpers.
- Use Pydantic attribute docstrings on models that emit JSON schema documentation, enabled by `BaseModelWithDocstrings` in `src/sol_execbench/core/data/base_model.py`.

## Function Design

**Size:** Keep new functions focused on one transformation, validation, probe, or CLI operation. Large orchestrators exist for CLI/evidence workflows in `src/sol_execbench/cli/main.py`, but new logic should generally be factored into testable private helpers.

**Parameters:** Use explicit typed parameters and keyword-only parameters when injecting test seams or optional behavior, such as `runner`, `which`, `timeout_seconds`, and `now` in `src/sol_execbench/core/environment.py`.

**Return Values:** Prefer typed domain objects, Pydantic models, dataclasses, tuples, or `Path | None` over loosely shaped dictionaries. Serialization functions should convert at the boundary with `model_dump(mode="json")` or JSON helpers, as in `src/sol_execbench/cli/main.py`.

## Module Design

**Exports:** Keep modules organized by domain: schemas under `src/sol_execbench/core/data/`, benchmark runtime helpers under `src/sol_execbench/core/bench/`, dataset helpers under `src/sol_execbench/core/dataset/`, scoring under `src/sol_execbench/core/scoring/`, CLI under `src/sol_execbench/cli/`, and generated driver templates under `src/sol_execbench/driver/templates/`.

**Barrel Files:** `src/sol_execbench/core/__init__.py` and package `__init__.py` files provide selected public imports. Prefer direct module imports for implementation-specific helpers and tests that target private behavior.

---

*Convention analysis: 2026-05-31*
