# Coding Conventions

**Analysis Date:** 2026-05-28

## Naming Patterns

**Files:**
- Use lowercase `snake_case.py` for source and test modules: `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/data/solution.py`, `tests/sol_execbench/test_rocm_profiler.py`.
- Keep package directories semantic and layered: `src/sol_execbench/core/data/`, `src/sol_execbench/core/bench/`, `src/sol_execbench/core/scoring/`, `src/sol_execbench/driver/`.
- Test files use `test_*.py` and mirror the source area when coverage is package-local: `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/driver/test_problem_packager.py`.

**Functions:**
- Use `snake_case` for public helpers and private module helpers: `time_runnable` in `src/sol_execbench/core/bench/timing.py`, `_write_environment_snapshot_sidecar` in `src/sol_execbench/cli/main.py`.
- Prefix internal validators and helper functions with `_`: `_validate_languages` in `src/sol_execbench/core/data/solution.py`, `_first_gfx_target` in `src/sol_execbench/driver/problem_packager.py`.
- Test functions use descriptive `test_*` names that encode expected behavior: `test_legacy_cuda_nvidia_languages_rejected_with_guidance` in `tests/sol_execbench/core/data/test_solution.py`.

**Variables:**
- Use `snake_case` for local variables and parameters: `output_directory`, `timeout_seconds`, `sidecar_path` in `src/sol_execbench/core/bench/rocm_profiler.py` and `src/sol_execbench/cli/main.py`.
- Use uppercase constants for module-level configuration and schemas: `ROCPROFV3_EXECUTABLE` in `src/sol_execbench/core/bench/rocm_profiler.py`, `ENV_SNAPSHOT_ENABLE_ENV` in `src/sol_execbench/cli/main.py`.
- Use typed collection aliases for JSON-like values where tests need casts: `JsonDict` in `tests/sol_execbench_type_helpers.py`.

**Types:**
- Use `PascalCase` for classes, dataclasses, Pydantic models, and enums: `Solution`, `BuildSpec`, `Rocprofv3ProfileResult`, `DiagnosticStage`.
- Enum classes subclass `str, Enum` when serialized into public schemas: `SupportedLanguages` in `src/sol_execbench/core/data/solution.py`, `DiagnosticStage` in `src/sol_execbench/core/diagnostics.py`.
- Use `Annotated` aliases for reusable Pydantic constraints: `NonEmptyString`, `NonNegativeInt` in `src/sol_execbench/core/data/base_model.py`.

## Code Style

**Formatting:**
- Tool: Ruff, configured in `pyproject.toml`.
- Use Python 3.12+ syntax and type unions with `|`: `Path | None` in `src/sol_execbench/cli/main.py`, `str | None` in `src/sol_execbench/core/diagnostics.py`.
- Add `from __future__ import annotations` in modules that use forward references or modern annotations: `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/bench/timing.py`, `tests/sol_execbench/test_e2e.py`.
- Keep SPDX license headers on source and test files when adding package code: `src/sol_execbench/core/data/solution.py`, `tests/conftest.py`.

**Linting:**
- Tool: Ruff from `pyproject.toml`.
- Key config: `pyproject.toml` excludes `.git`, `.venv`, `__pycache__`, `build`, `dist`, `*.egg-info`, `.ruff_cache`, `data`, and `examples`.
- Key lint exception: `E741` is ignored in `pyproject.toml`.
- Type checking uses `ty` over `src` and `tests` via `[tool.ty.src]` in `pyproject.toml`.

## Import Organization

**Order:**
1. Future imports: `from __future__ import annotations`.
2. Standard library imports: `json`, `subprocess`, `dataclasses`, `pathlib`, `collections.abc`.
3. Third-party imports: `click`, `pytest`, `torch`, `pydantic`, `rich`.
4. First-party imports: `sol_execbench.core...` or package-relative imports inside source modules.

**Path Aliases:**
- No configured import alias is detected in `pyproject.toml`.
- Source modules use relative imports inside package boundaries when nearby: `from ..core import ...` in `src/sol_execbench/driver/problem_packager.py`, `from .base_model import ...` in `src/sol_execbench/core/data/solution.py`.
- Tests use absolute package imports from the installed `src` package: `from sol_execbench.core.data.solution import BuildSpec` in `tests/sol_execbench/core/data/test_solution.py`.

## Error Handling

**Patterns:**
- Raise `ValueError` for schema, validation, and invalid argument conditions: validators in `src/sol_execbench/core/data/solution.py`, timing mode checks in `src/sol_execbench/core/bench/timing.py`.
- Raise `click.ClickException` for user-facing CLI argument and mode errors: `_resolve_problem_dir` and CLI subcommands in `src/sol_execbench/cli/main.py`.
- Use `SolExecBenchError` for stage-aware internal diagnostics with remediation hints: `src/sol_execbench/core/diagnostics.py`.
- Make optional diagnostic evidence nonfatal by catching broad exceptions at sidecar boundaries and printing warnings: `_write_environment_snapshot_sidecar`, `_write_profile_sidecar`, `_write_static_evidence_sidecar` in `src/sol_execbench/cli/main.py`.
- Preserve original exceptions with `from exc` when reclassifying parse failures: `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/scoring/baseline_artifact.py`.

## Logging

**Framework:** Rich console for CLI output; direct return metadata for library code.

**Patterns:**
- Use `Console(stderr=True)` and `console.print(...)` for CLI status, warnings, tables, and failure output in `src/sol_execbench/cli/main.py`.
- Keep core modules mostly side-effect free; return dataclasses, dictionaries, or diagnostics instead of printing: `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/diagnostics.py`.
- Tests assert returned payloads and sidecar contents rather than terminal output when possible: `tests/sol_execbench/test_cli_environment_snapshot.py`, `tests/sol_execbench/test_rocm_profiler.py`.

## Comments

**When to Comment:**
- Use module docstrings to state purpose and boundary behavior: `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/driver/problem_packager.py`.
- Use docstrings on public classes, public helpers, validators, and fixtures where behavior matters: `Solution` in `src/sol_execbench/core/data/solution.py`, `tmp_cache_dir` in `tests/conftest.py`.
- Use brief inline comments for benchmark semantics, ROCm compatibility, and non-obvious safety decisions: `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`.

**JSDoc/TSDoc:**
- Not applicable; this is a Python codebase.
- Pydantic models use attribute docstrings to populate schema documentation through `BaseModelWithDocstrings` in `src/sol_execbench/core/data/base_model.py`.

## Function Design

**Size:** Keep simple pure helpers small and focused; larger orchestration functions are isolated at CLI and driver boundaries such as `src/sol_execbench/cli/main.py` and `src/sol_execbench/driver/templates/eval_driver.py`.

**Parameters:** Prefer explicit typed parameters and dependency injection for IO boundaries. Examples include injectable `collector` in `_write_environment_snapshot_sidecar` (`src/sol_execbench/cli/main.py`) and injectable `runner` in `collect_rocprofv3_profile` (`src/sol_execbench/core/bench/rocm_profiler.py`).

**Return Values:** Prefer typed dataclasses, Pydantic models, tuples, and JSON-serializable dictionaries. Examples include `Rocprofv3ProfileResult.to_dict()` in `src/sol_execbench/core/bench/rocm_profiler.py`, `ProblemPackager.compile()` returning `(cmd, artifact_path)` in `src/sol_execbench/driver/problem_packager.py`.

## Module Design

**Exports:** Core package exports are centralized through package `__init__.py` modules such as `src/sol_execbench/core/__init__.py` and `src/sol_execbench/core/data/__init__.py`; import from these surfaces when they already expose the needed model.

**Barrel Files:** Barrel files are used for stable public surfaces in `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/data/__init__.py`, and `src/sol_execbench/driver/__init__.py`. Avoid adding unrelated symbols to these files unless the symbol is part of the public package API.

---

*Convention analysis: 2026-05-28*
