# Coding Conventions

**Analysis Date:** 2026-06-01

## Naming Patterns

**Files:**
- Use lowercase `snake_case.py` for source modules under `src/sol_execbench/`, such as `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, and `src/sol_execbench/core/bench/clock_lock.py`.
- Use `test_*.py` for pytest modules under `tests/`, mirroring the package path when practical, such as `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/core/bench/test_clock_lock.py`, and `tests/sol_execbench/driver/test_eval_driver.py`.
- Keep generated driver templates in `src/sol_execbench/driver/templates/`, with executable-style module names like `src/sol_execbench/driver/templates/eval_driver.py` and `src/sol_execbench/driver/templates/build_ext.py`.

**Functions:**
- Use `snake_case` for public helpers and internal helpers, such as `build_cli_command()` in `src/sol_execbench/core/dataset/runner.py`, `collect_rocprofv3_profile()` in `src/sol_execbench/core/bench/rocm_profiler.py`, and `_load_solution()` in `src/sol_execbench/cli/main.py`.
- Prefix private module helpers with `_`, as in `_validate_compile_flag()` in `src/sol_execbench/core/data/solution.py`, `_rocm_smi_executable()` in `src/sol_execbench/core/bench/clock_lock.py`, and `_run_eval_driver_process()` in `tests/sol_execbench/driver/test_eval_driver.py`.
- Use `test_<expected_behavior>` names for tests. Prefer behavior-specific names such as `test_legacy_cuda_nvidia_languages_rejected_with_guidance` in `tests/sol_execbench/core/data/test_solution.py`.

**Variables:**
- Use `snake_case` for local variables and parameters, such as `definition_path`, `workload_path`, `solution_path`, and `output_dir` in `src/sol_execbench/core/dataset/runner.py`.
- Use uppercase names for module constants, such as `ROCPROFV3_EXECUTABLE` in `src/sol_execbench/core/bench/rocm_profiler.py`, `_DIAGNOSTIC_TAIL_LIMIT` in `src/sol_execbench/cli/main.py`, and `CLI_LOG_LIMIT` in `src/sol_execbench/core/dataset/runner.py`.
- Use leading underscores for private module constants, such as `_PATH_INJECTION_PREFIXES` in `src/sol_execbench/core/data/solution.py` and `_MODULE` in `tests/sol_execbench/core/bench/test_clock_lock.py`.

**Types:**
- Use `PascalCase` for classes, Pydantic models, dataclasses, and enums, such as `BuildSpec`, `SourceFile`, and `SupportedLanguages` in `src/sol_execbench/core/data/solution.py`.
- Use uppercase enum members with string values for schema-facing enums, such as `SupportedLanguages.HIP_CPP = "hip_cpp"` and `SupportedHardware.GFX1200 = "gfx1200"` in `src/sol_execbench/core/data/solution.py`.
- Use explicit type aliases for repeated structural types, such as `NonEmptyString` and `NonNegativeInt` in `src/sol_execbench/core/data/base_model.py`, `ProfilerRunner` in `src/sol_execbench/core/bench/rocm_profiler.py`, and `JsonDict` in `tests/sol_execbench_type_helpers.py`.

## Code Style

**Formatting:**
- Ruff is the formatting and linting authority via `[tool.ruff]` in `pyproject.toml`.
- Use Python `>=3.12,<3.14` syntax as declared in `pyproject.toml`, including `list[str]`, `dict[str, Any]`, `Path | None`, and `from __future__ import annotations` in most non-trivial modules.
- Keep files ASCII unless an existing file already uses a specific symbol. The repository contains some docstrings/messages with non-ASCII arrows in `src/sol_execbench/core/data/definition.py`; new code should prefer ASCII unless matching nearby text.
- Preserve SPDX license headers in source and test files that already use them, such as `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/data/solution.py`, and `tests/conftest.py`.
- Exclude generated or external-heavy areas from formatting/linting according to `pyproject.toml`: `data`, `examples`, build output, virtualenvs, and caches.

**Linting:**
- Ruff is configured in `pyproject.toml`; only `E741` is globally ignored.
- Type checking uses `ty` with `[tool.ty.src] include = ["src", "tests"]` in `pyproject.toml`.
- Run lint with `uv run --with ruff ruff check .` and format with `uv run --with ruff ruff format .`.
- Avoid broad style refactors in focused changes; follow nearby layout in files such as `src/sol_execbench/core/scoring/amd_sol_v2.py` and `src/sol_execbench/core/dataset/runner.py`.

## Import Organization

**Order:**
1. `from __future__ import annotations` when needed, immediately after the module docstring, as in `src/sol_execbench/cli/main.py` and `src/sol_execbench/core/bench/rocm_profiler.py`.
2. Standard library imports, grouped together, such as `json`, `subprocess`, `Path`, and `typing` imports in `src/sol_execbench/core/dataset/runner.py`.
3. Third-party imports, such as `click`, `rich`, `pydantic`, `pytest`, and `torch` in `src/sol_execbench/cli/main.py` and `tests/sol_execbench/driver/test_eval_driver.py`.
4. First-party absolute imports from `sol_execbench.*` in tests and cross-package modules, such as `from sol_execbench.core.dataset import runner` in `tests/sol_execbench/test_dataset_runner.py`.
5. Intra-package relative imports in package modules when importing siblings, such as `from .base_model import BaseModelWithDocstrings` in `src/sol_execbench/core/data/solution.py`.

**Path Aliases:**
- No custom import alias is configured in `pyproject.toml`.
- Tests import package code through the installed `src` layout as `sol_execbench.*`, such as `from sol_execbench.core.data.solution import BuildSpec` in `tests/sol_execbench/core/data/test_solution.py`.
- Shared test construction helpers live in `tests/sol_execbench_type_helpers.py`; use `make_definition()`, `make_solution()`, `make_workload()`, `make_build_spec()`, and `make_trace()` instead of repeating raw `model_validate()` calls in tests.

## Error Handling

**Patterns:**
- Use `ValueError` for invalid domain data, schema migration violations, unsafe paths, unsupported flags, and invalid report content. Examples include `_validate_compile_flag()` in `src/sol_execbench/core/data/solution.py`, `Definition` validators in `src/sol_execbench/core/data/definition.py`, and `build_dataset_inventory()` helpers in `src/sol_execbench/core/dataset/inventory.py`.
- Chain exceptions when preserving parse or validation context, such as `raise ValueError(... ) from exc` in `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/matrix_diff.py`, and `src/sol_execbench/core/data/definition.py`.
- Return explicit status objects or `None` for recoverable subprocess/reporting failures where callers need to continue, such as `run_cli()` returning `None` after logging failures in `src/sol_execbench/core/dataset/runner.py`.
- Use `click.ClickException` for user-facing CLI validation errors in `src/sol_execbench/cli/main.py`.
- Catch narrow exceptions at system boundaries where possible: `FileNotFoundError`, `subprocess.CalledProcessError`, `subprocess.TimeoutExpired`, `json.JSONDecodeError`, `OSError`, `RuntimeError`, and `AttributeError` are used throughout `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/environment.py`, and `src/sol_execbench/core/dataset/runner.py`.
- Avoid bare `except`. Existing broad `except Exception` blocks are used at diagnostic boundaries that must not break primary evaluation, such as optional environment snapshot, profiling, and static evidence sidecar collection in `src/sol_execbench/cli/main.py`.

## Logging

**Framework:** `logging` for library internals; Rich `Console` and `print()` for CLI/script output.

**Patterns:**
- Use `logging.getLogger(__name__)` in reusable library modules that need runtime diagnostics, such as `src/sol_execbench/core/bench/clock_lock.py`.
- Use `logger.warning()` and `logger.info()` for recoverable hardware/tooling state in library code, such as failed ROCm clock locks in `src/sol_execbench/core/bench/clock_lock.py`.
- Use a Rich `Console(stderr=True)` for interactive CLI status and warnings in `src/sol_execbench/cli/main.py`.
- Use `print()` only in command-style modules and script helpers that write CLI JSON or human-readable reports, such as `src/sol_execbench/core/dataset/runner.py`, `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/runtime_evidence.py`, and `src/sol_execbench/core/dependency_matrix.py`.
- Bound or truncate captured subprocess output before persisting diagnostics. Examples include `_diagnostic_tail()` in `src/sol_execbench/cli/main.py` and `bounded_cli_stream()` in `src/sol_execbench/core/dataset/runner.py`.

## Comments

**When to Comment:**
- Use module docstrings to state the file responsibility, as in `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and `tests/sol_execbench/driver/test_eval_driver.py`.
- Use class and function docstrings for public helpers, schema models, validators, and test helpers. Examples include `Rocprofv3TimingEvidence` in `src/sol_execbench/core/bench/rocm_profiler.py` and `_run_eval_driver()` in `tests/sol_execbench/driver/test_eval_driver.py`.
- Use short comments to label test phases or clarify invariants, such as the compile/evaluate phase comments in `tests/examples/test_examples.py`.
- Avoid comments that merely restate assignments; prefer comments for security boundaries, subprocess behavior, hardware gating, and schema compatibility constraints.

**JSDoc/TSDoc:**
- Not applicable. This repository uses Python docstrings, Pydantic attribute docstrings, and inline comments.
- Pydantic schema models should inherit `BaseModelWithDocstrings` from `src/sol_execbench/core/data/base_model.py` when JSON schema attribute descriptions matter.

## Function Design

**Size:** Keep unit helpers small and pure when possible. Place orchestration in explicit higher-level functions such as `run_cli()` in `src/sol_execbench/core/dataset/runner.py` and CLI commands in `src/sol_execbench/cli/main.py`.

**Parameters:** Prefer keyword-only parameters for multi-argument orchestration helpers, especially path-heavy helpers like `build_cli_command()` and `run_cli()` in `src/sol_execbench/core/dataset/runner.py`.

**Return Values:** Prefer typed return values that make failure modes explicit: `list[dict] | None` in `run_cli()`, frozen dataclasses with `to_dict()` methods in `src/sol_execbench/core/bench/rocm_profiler.py`, Pydantic models for schemas under `src/sol_execbench/core/data/`, and `bool` for probe/verification helpers in `src/sol_execbench/core/bench/clock_lock.py`.

## Module Design

**Exports:** Package `__init__.py` files re-export public domain objects for stable imports, such as `src/sol_execbench/core/__init__.py` and `src/sol_execbench/driver/__init__.py`. Use explicit imports from concrete modules when working near implementation details.

**Barrel Files:** Use package barrels sparingly for public surfaces. Tests often import concrete modules directly for focused behavior, such as `sol_execbench.core.bench.clock_lock` in `tests/sol_execbench/core/bench/test_clock_lock.py`.

---

*Convention analysis: 2026-06-01*
