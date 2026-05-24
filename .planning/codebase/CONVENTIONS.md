# Coding Conventions

**Analysis Date:** 2026-05-24

## Naming Patterns

**Files:**
- Use `snake_case.py` for source and tests: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/data/solution.py`, `tests/sol_execbench/test_rocm_profiler.py`.
- Keep tests under the package area they exercise when possible: `tests/sol_execbench/core/data/test_solution.py` mirrors `src/sol_execbench/core/data/solution.py`; `tests/sol_execbench/driver/test_build_ext.py` mirrors `src/sol_execbench/driver/templates/build_ext.py`.
- Template scripts live in `src/sol_execbench/driver/templates/` and are validated as executable Python by tests in `tests/sol_execbench/driver/`.

**Functions:**
- Use `snake_case` for public and private functions: `bench_time_with_device_events()` in `src/sol_execbench/core/bench/timing.py`, `review_solution_sources()` in `src/sol_execbench/core/bench/reward_hack.py`, `_resolve_problem_dir()` in `src/sol_execbench/cli/main.py`.
- Prefix module-private helpers with `_`: `_summarize_statistics()` in `src/sol_execbench/core/bench/timing.py`, `_match_rule()` in `src/sol_execbench/core/bench/reward_hack.py`, `_load_sample()` in `tests/sol_execbench/test_e2e.py`.
- Use descriptive validator names on Pydantic models: `_validate_entry_point()`, `_validate_languages()`, and `_reject_legacy_compile_options()` in `src/sol_execbench/core/data/solution.py`.

**Variables:**
- Use `snake_case` for local variables and fields: `compile_timeout`, `output_file`, and `keep_staging` in `src/sol_execbench/cli/main.py`.
- Use uppercase module constants for immutable fixtures, schemas, and lookup sets: `REQUIRED_CAPABILITIES` in `tests/sol_execbench/test_contract.py`, `_STATIC_RULES` in `src/sol_execbench/core/bench/reward_hack.py`, `_CPP_LANGUAGES` in `tests/sol_execbench/test_e2e.py`.
- Use leading underscores for module-local constants that should not be imported as API: `_TEMPLATES_DIR` in `tests/sol_execbench/driver/test_eval_driver.py`, `_ELAPSED_TIME_ADDR` in `src/sol_execbench/core/bench/reward_hack.py`.

**Types:**
- Use `PascalCase` for classes, dataclasses, Pydantic models, and enums: `Definition` in `src/sol_execbench/core/data/definition.py`, `SourceReviewIssue` in `src/sol_execbench/core/bench/reward_hack.py`, `SupportedLanguages` in `src/sol_execbench/core/data/solution.py`.
- Enum classes use `PascalCase`; enum members use uppercase names with string values: `SupportedLanguages.HIP_CPP = "hip_cpp"` in `src/sol_execbench/core/data/solution.py`.
- Prefer Pydantic models for public JSON schemas and validation boundaries: `Definition`, `Workload`, `Solution`, and `Trace` under `src/sol_execbench/core/data/`.
- Prefer `@dataclass(frozen=True)` for internal immutable records: `SourceReviewIssue` in `src/sol_execbench/core/bench/reward_hack.py`, `Rocprofv3CollectionRequest` in `src/sol_execbench/core/bench/rocm_profiler.py`, and scoring records in `src/sol_execbench/core/scoring/amd_score.py`.

## Code Style

**Formatting:**
- Use Ruff formatting from `pyproject.toml`; no separate `.prettierrc`, `.eslintrc`, `ruff.toml`, `pytest.ini`, or `.coveragerc` is present.
- Keep Python compatible with `requires-python = ">=3.12,<3.14"` in `pyproject.toml`.
- Keep line wrapping and import formatting consistent with Ruff output. Existing code uses parenthesized multiline imports in `src/sol_execbench/core/__init__.py` and multiline assertions in tests such as `tests/sol_execbench/test_e2e.py`.
- Preserve SPDX and Apache license headers in source and test files that already use them, such as `src/sol_execbench/core/data/definition.py` and `tests/conftest.py`.

**Linting:**
- Ruff is configured in `pyproject.toml`.
- Excluded paths are `.git`, `.venv`, `__pycache__`, `build`, `dist`, `*.egg-info`, `.ruff_cache`, `data`, and `examples`.
- Ruff lint ignores `E741`; avoid adding ambiguous single-character names anyway unless matching nearby math-heavy code.
- Run lint with `uv run --with ruff ruff check .`; run formatting with `uv run --with ruff ruff format .`.

## Import Organization

**Order:**
1. `from __future__ import annotations` where modern annotations are used, as in `src/sol_execbench/core/bench/timing.py` and `tests/sol_execbench/test_e2e.py`.
2. Standard library imports: `json`, `subprocess`, `Path`, `dataclass`, `Enum`.
3. Third-party imports: `click`, `pytest`, `pydantic`, `torch`, `rich`.
4. Package imports from `sol_execbench` or relative imports inside the package.
5. `TYPE_CHECKING` imports stay behind `if TYPE_CHECKING:` as in `src/sol_execbench/core/data/definition.py`.

**Path Aliases:**
- No configured path aliases are detected. Use normal package imports rooted at `sol_execbench`, for example `from sol_execbench.core.data.solution import BuildSpec` in `tests/sol_execbench/core/data/test_solution.py`.
- Inside package modules, short relative imports are common within the same layer: `from .base_model import BaseModelWithDocstrings` in `src/sol_execbench/core/data/definition.py`.
- Public re-export modules use explicit `__all__` lists: `src/sol_execbench/__init__.py`, `src/sol_execbench/core/__init__.py`, and `src/sol_execbench/core/scoring/__init__.py`.

## Error Handling

**Patterns:**
- Raise `ValueError` for schema, argument, and semantic validation failures: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/bench/timing.py`, and `src/sol_execbench/core/scoring/amd_hardware_models.py`.
- Chain parsing and conversion exceptions with `from exc` when preserving cause matters, as in `src/sol_execbench/core/baseline.py` and `src/sol_execbench/core/scoring/amd_hardware_models.py`.
- Use domain-specific exception classes for security and benchmark integrity failures: `RewardHackDetected` in `src/sol_execbench/core/bench/reward_hack.py`.
- Use `click.ClickException` for user-facing CLI validation in `src/sol_execbench/cli/main.py`.
- Use structured failure status values rather than ad hoc strings across trace data: `EvaluationStatus` in `src/sol_execbench/core/data/trace.py`.
- Suppress broad exceptions only at explicit environment-probe or best-effort diagnostic boundaries, such as ROCm availability checks in `tests/conftest.py` and utility probes in `src/sol_execbench/core/utils.py`.

## Logging

**Framework:** `logging`, Rich console output, subprocess stdout/stderr capture.

**Patterns:**
- Use module loggers for library code that needs logs: `logger = logging.getLogger(__name__)` in `src/sol_execbench/core/bench/clock_lock.py`.
- Use Rich `Console(stderr=True)` for CLI status and result presentation in `src/sol_execbench/cli/main.py`.
- Use plain `print()` only in standalone generated/executed scripts or JSON output paths where stdout is part of the contract: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`, and the `--json` branch in `src/sol_execbench/cli/main.py`.
- Tests capture subprocess output and include `stdout`/`stderr` in assertion messages for debuggability, as in `tests/sol_execbench/test_e2e.py` and `tests/sol_execbench/driver/test_eval_driver.py`.

## Comments

**When to Comment:**
- Use module docstrings to define purpose and execution context: `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/bench/reward_hack.py`, and `tests/sol_execbench/driver/test_eval_driver.py`.
- Use comments to identify phases, security boundaries, and non-obvious ROCm compatibility behavior: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/reward_hack.py`, and `tests/sol_execbench/test_e2e.py`.
- Avoid comments that restate obvious code. Prefer comments before complex benchmark, subprocess, or security-sensitive blocks.

**JSDoc/TSDoc:**
- Not applicable; no TypeScript or JavaScript source is detected.
- Python docstrings are common on public classes/functions and many test helpers. Use NumPy-style sections (`Parameters`, `Returns`, `Raises`) where surrounding code does, such as `src/sol_execbench/core/data/definition.py`.
- Pydantic schema fields often use attribute docstrings through `BaseModelWithDocstrings` in `src/sol_execbench/core/data/base_model.py`; follow this pattern for new schema fields.

## Function Design

**Size:** Keep helpers focused around one validation, conversion, or IO boundary. Larger modules such as `src/sol_execbench/core/scoring/solar_derivation.py` and `src/sol_execbench/core/scoring/amd_bound_graph.py` use many small private helpers to isolate parsing, classification, and validation substeps.

**Parameters:** Prefer typed parameters and explicit return types for production code. Use `Path` for filesystem paths, `Callable` for injectable runners, and `Literal` for constrained string options, as in `src/sol_execbench/core/bench/timing.py` and `src/sol_execbench/core/bench/rocm_profiler.py`.

**Return Values:** Return structured data instead of loosely shaped dictionaries at public boundaries. Use Pydantic models for public JSON contracts (`src/sol_execbench/core/data/trace.py`), dataclasses for internal records (`src/sol_execbench/core/bench/rocm_profiler.py`), and plain dictionaries only at serialization or fixture boundaries.

## Module Design

**Exports:** Package boundary modules should import and list intended public symbols in `__all__`, as in `src/sol_execbench/core/__init__.py` and `src/sol_execbench/cli/__init__.py`.

**Barrel Files:** Barrel files are used sparingly for public API convenience. Keep implementation logic in leaf modules such as `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/data/solution.py`, and `src/sol_execbench/core/scoring/amd_score.py`; expose selected symbols through package `__init__.py` files.

---

*Convention analysis: 2026-05-24*
