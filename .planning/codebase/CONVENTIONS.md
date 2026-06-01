# Coding Conventions

**Analysis Date:** 2026-06-01

## Naming Patterns

**Files:**
- Use lowercase `snake_case.py` modules for package code under `src/sol_execbench/`, such as `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/bench/eval_runtime.py`, and `src/sol_execbench/driver/problem_packager.py`.
- Keep test filenames aligned with the module or workflow under test using `test_*.py`, such as `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/driver/test_problem_packager.py`, and `tests/examples/test_examples.py`.
- Keep generated driver templates in `src/sol_execbench/driver/templates/` and test them through template-loading tests such as `tests/sol_execbench/driver/test_eval_driver.py` and `tests/sol_execbench/driver/test_build_ext.py`.
- Keep sample fixtures and benchmark examples as JSON/Python source under `tests/sol_execbench/samples/`, `tests/sol_execbench/fixtures/`, and `examples/`; these paths are intentionally excluded from Ruff for generated or benchmark payloads in `pyproject.toml`.

**Functions:**
- Use `snake_case` for functions and methods, including private helpers like `_load_solution` in `src/sol_execbench/cli/main.py`, `_get_local_gfx` in `src/sol_execbench/driver/problem_packager.py`, and `infer_destination_passing_style` in `src/sol_execbench/core/dataset/runner.py`.
- Prefix module-private helpers with `_` when they are implementation details, such as `_missing_rocm_device_nodes` in `tests/conftest.py`, `_cpu_time_runnable` in `src/sol_execbench/core/bench/eval_runtime.py`, and `_inject_offload_arch_flags` in `src/sol_execbench/driver/problem_packager.py`.
- Prefer small, composable helper functions around subprocess, file, and JSON boundaries, for example `build_cli_command`, `bounded_cli_stream`, and `save_cli_log` in `src/sol_execbench/core/dataset/runner.py`.

**Variables:**
- Use `snake_case` for locals and attributes, such as `definition_path`, `workload_path`, `solution_path`, `output_dir`, and `keep_staging` in `src/sol_execbench/core/dataset/runner.py`.
- Use uppercase module constants for fixed configuration and enumerated sets, such as `NATIVE_ROCM_LANGUAGES` in `src/sol_execbench/core/bench/eval_runtime.py`, `_CPP_LANGUAGES` in `src/sol_execbench/driver/problem_packager.py`, and `ENV_SNAPSHOT_ENABLE_ENV` in `src/sol_execbench/cli/main.py`.
- Use explicit type annotations on module-level collections and public helper returns when they clarify schema shape, such as `JsonDict = dict[str, Any]` in `tests/sol_execbench_type_helpers.py` and `PathExists = Callable[[Path], bool]` in `tests/conftest.py`.

**Types:**
- Use `PascalCase` for dataclasses, Pydantic models, and enums, such as `ReferenceTimingResult` in `src/sol_execbench/core/bench/eval_runtime.py`, `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py`, and `SupportedLanguages` in `src/sol_execbench/core/data/solution.py`.
- Use `str, Enum` for schema enums so serialized values are stable strings, as in `SupportedLanguages`, `SupportedHardware`, and `SupportedBindings` in `src/sol_execbench/core/data/solution.py`.
- Use Pydantic v2 models for public benchmark schemas and keep validators on the model that owns the contract, such as `BuildSpec._reject_legacy_languages` and `SourceFile._validate_source_path` in `src/sol_execbench/core/data/solution.py`.
- Use frozen dataclasses for simple immutable result objects that are not public JSON schema models, such as `TimingResult` and `ReferenceTimingResult` in `src/sol_execbench/core/bench/eval_runtime.py`.

## Code Style

**Formatting:**
- Ruff is the formatter and linter configured in `pyproject.toml`; run `uv run --with ruff ruff format .` and `uv run --with ruff ruff check .`.
- Use Python 3.12+ syntax. Built-in generics like `list[str]`, `dict[str, Any]`, union syntax like `Path | None`, and `from __future__ import annotations` are standard in files such as `src/sol_execbench/core/dataset/runner.py` and `src/sol_execbench/driver/problem_packager.py`.
- Keep line wrapping compatible with Ruff defaults. Long assertions and error messages are wrapped with parentheses, as in `tests/sol_execbench/driver/test_eval_driver.py` and `tests/sol_execbench/test_e2e.py`.
- Preserve SPDX headers in source and test files that carry retained upstream licensing, such as `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/data/solution.py`, and `tests/sol_execbench/driver/test_problem_packager.py`.

**Linting:**
- Ruff configuration lives in `pyproject.toml`; generated and data-heavy paths are excluded: `.git`, `.venv`, `__pycache__`, `build`, `dist`, `*.egg-info`, `.ruff_cache`, `data`, and `examples`.
- Ruff ignores `E741` in `pyproject.toml`, allowing legacy single-letter ambiguous names where nearby code or math notation already uses them.
- Type checking uses `ty` with `src` and `tests` included in `pyproject.toml`; run `uv run ty check`.
- CI enforces `uv run ruff check .`, `uv run ty check`, and selected pytest commands in `.github/workflows/code-quality.yml`.

## Import Organization

**Order:**
1. `from __future__ import annotations` when needed, immediately after the module docstring or license header, as in `src/sol_execbench/core/bench/eval_runtime.py`.
2. Standard library imports, grouped alphabetically enough to scan, such as `json`, `subprocess`, `sys`, `dataclasses`, `pathlib`, and `typing` in `src/sol_execbench/cli/main.py`.
3. Third-party imports such as `click`, `rich`, `pytest`, `pydantic`, and `torch`, as shown in `src/sol_execbench/cli/main.py`, `tests/conftest.py`, and `tests/sol_execbench/driver/test_eval_driver.py`.
4. Local package imports from `sol_execbench...` or relative imports from sibling modules, as in `src/sol_execbench/driver/problem_packager.py`.

**Path Aliases:**
- No custom import alias is configured in `pyproject.toml`; import package code through the installed `sol_execbench` package rooted at `src/sol_execbench/`.
- Test helpers are imported directly from `tests/sol_execbench_type_helpers.py`, for example `from sol_execbench_type_helpers import make_build_spec` in `tests/sol_execbench/core/data/test_solution.py`.
- Scripts that need to import a repository script use `importlib.util.spec_from_file_location`, as in `tests/sol_execbench/test_run_dataset_execution_closure.py` for `scripts/run_dataset.py`.

## Error Handling

**Patterns:**
- Raise Click-facing errors as `click.ClickException` in CLI validation paths, such as `_resolve_problem_dir` in `src/sol_execbench/cli/main.py`.
- Raise `ValueError` from Pydantic validators for schema violations and include ROCm migration guidance in the message, as in `BuildSpec._reject_legacy_languages` and `BuildSpec._reject_legacy_compile_options` in `src/sol_execbench/core/data/solution.py`.
- Raise `RuntimeError` when staged evaluation input is malformed or dynamic imports fail, as in `load_staged_problem` and `load_reference_function` in `src/sol_execbench/core/bench/eval_runtime.py`.
- Return structured failure result objects when runtime timing should remain reportable, as in `measure_latency` and `measure_reference_latency` in `src/sol_execbench/core/bench/eval_runtime.py`.
- Catch subprocess `TimeoutExpired` and non-zero exits at dataset-run boundaries, write bounded logs, and return `None`, as in `run_cli`, `save_cli_log`, and `bounded_cli_stream` in `src/sol_execbench/core/dataset/runner.py`.
- Use `assert` for internal invariants inside implementation code where invalid call order indicates programmer error, such as `ProblemPackager.compile()` in `src/sol_execbench/driver/problem_packager.py`.

## Logging

**Framework:** Rich console for CLI, plain captured stdout/stderr for subprocess workflows.

**Patterns:**
- Use `Console(stderr=True)` for user-facing CLI presentation in `src/sol_execbench/cli/main.py`; keep machine-readable trace JSON on stdout.
- Render tables with Rich `Table` for human CLI summaries, as in `_print_traces_table` in `src/sol_execbench/cli/main.py`.
- Keep user solution stdout out of trace JSONL; tests assert this in `tests/sol_execbench/driver/test_eval_driver.py`.
- For dataset scripts, print high-level progress and write bounded failure logs to files, as in `run_cli`, `save_cli_log`, and `save_cli_timeout_log` in `src/sol_execbench/core/dataset/runner.py`.
- Avoid logging secrets or full environment payloads. Environment and closure tests verify bounded/sanitized provenance in `tests/sol_execbench/test_run_dataset_execution_closure.py`.

## Comments

**When to Comment:**
- Use module docstrings to state workflow boundaries and intended execution mode, as in `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/bench/eval_runtime.py`, and `tests/sol_execbench/driver/test_eval_driver.py`.
- Use short comments to identify phases in longer procedural tests or CLI flows, such as compile/evaluate phases in `tests/sol_execbench/test_e2e.py`.
- Use comments for non-obvious migration constraints and security boundaries, such as path traversal validation in `SourceFile` in `src/sol_execbench/core/data/solution.py`.
- Keep comments close to behavior and avoid restating simple assignments.

**JSDoc/TSDoc:**
- Not applicable; this repository is Python.
- Use Python docstrings for public classes, models, and helpers. Pydantic model attribute docstrings are part of schema generation through `BaseModelWithDocstrings` in `src/sol_execbench/core/data/base_model.py`.

## Function Design

**Size:** Prefer focused helpers for parsing, validation, subprocess assembly, and result conversion. Larger orchestration functions belong at CLI/script boundaries such as `src/sol_execbench/cli/main.py` and `scripts/run_dataset.py`.

**Parameters:** Use keyword-only parameters for helpers with multiple path/config inputs, as in `build_cli_command` and `run_cli` in `src/sol_execbench/core/dataset/runner.py`. Use typed optional dependencies to support testing, such as `time_fn` in `measure_latency` and `path_exists` in `tests/conftest.py`.

**Return Values:** Return concrete domain types where possible, such as `Definition`, `Workload`, `Solution`, and `Trace` in `src/sol_execbench/cli/main.py` and `src/sol_execbench/driver/problem_packager.py`. Return tuples for small grouped outputs with documented ordering, such as `load_staged_problem` in `src/sol_execbench/core/bench/eval_runtime.py`.

## Module Design

**Exports:** Keep package exports explicit in `__init__.py` modules under `src/sol_execbench/`, and import concrete classes/functions from their owning module in tests when behavior is module-specific.

**Barrel Files:** Use package-level re-export modules sparingly for public core API access, such as imports from `sol_execbench.core` in `src/sol_execbench/cli/main.py` and `tests/sol_execbench/test_e2e.py`.

---

*Convention analysis: 2026-06-01*
