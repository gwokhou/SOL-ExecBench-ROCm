# Coding Conventions

**Analysis Date:** 2026-05-22

## Naming Patterns

**Files:**
- Use lowercase `snake_case.py` for Python source and test modules: `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/bench/clock_lock.py`, `tests/sol_execbench/core/data/test_solution.py`.
- Test modules use `test_*.py` and mirror the package path when they cover package internals: `tests/sol_execbench/driver/test_problem_packager.py` covers `src/sol_execbench/driver/problem_packager.py`.
- Template scripts that are copied/executed by the driver live under `src/sol_execbench/driver/templates/`: `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/templates/eval_driver.py`.
- Runnable scripts live under `scripts/` and use action names: `scripts/run_dataset.py`, `scripts/download_solexecbench.py`.

**Functions:**
- Use `snake_case` for public and private functions: `load_json_file()` in `src/sol_execbench/core/data/json_utils.py`, `probe_clock_lock_available()` in `src/sol_execbench/core/bench/clock_lock.py`.
- Prefix internal helpers with `_`: `_load_solution()` in `src/sol_execbench/cli/main.py`, `_run_subprocess()` in `tests/sol_execbench/test_e2e.py`, `_make_spec()` in `tests/sol_execbench/core/data/test_solution.py`.
- Validator methods use an underscore plus a verb phrase: `_validate_source_path()` and `_reject_legacy_languages()` in `src/sol_execbench/core/data/solution.py`.
- Keep CLI command functions named `cli` in command modules: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`.

**Variables:**
- Use `snake_case` for locals and parameters: `target_hardware`, `entry_point`, `compile_options` in `src/sol_execbench/core/data/solution.py`.
- Use uppercase for module constants and test data tables: `_CPP_LANGUAGES` in `tests/sol_execbench/test_e2e.py`, `VERIFY_DELAY_S` in `src/sol_execbench/core/bench/clock_lock.py`, `NATIVE_LANGUAGES` in `tests/sol_execbench/core/data/test_solution.py`.
- Prefix module-private constants with `_`: `_TEMPLATE_PATH` in `tests/sol_execbench/driver/test_build_ext.py`, `_SAMPLES_DIR` in `tests/sol_execbench/test_e2e.py`.

**Types:**
- Use `PascalCase` for classes, Pydantic models, dataclasses, and enums: `BuildSpec`, `SourceFile`, `SupportedLanguages` in `src/sol_execbench/core/data/solution.py`; `TraceRunSummary` in `src/sol_execbench/core/reporting.py`.
- Enum members use uppercase names with string values matching public schema values: `SupportedLanguages.HIP_CPP = "hip_cpp"` in `src/sol_execbench/core/data/solution.py`.
- Type aliases use `PascalCase`: `NonEmptyString`, `NonNegativeInt` in `src/sol_execbench/core/data/base_model.py`.

## Code Style

**Formatting:**
- Use Ruff formatting via `uv run ruff format .`; configuration is in `pyproject.toml`.
- Target Python is `>=3.12,<3.14` in `pyproject.toml`; use modern typing syntax such as `list[Workload]`, `dict[str, object]`, and `str | None`.
- Keep imports grouped as standard library, third-party packages, then local package imports. Examples: `src/sol_execbench/core/data/definition.py`, `tests/sol_execbench/driver/test_build_ext.py`.
- Include SPDX license headers on Python source and tests. Most files include the full Apache-2.0 header, while some test files use the two-line SPDX header, as in `tests/sol_execbench/core/data/test_solution.py`.

**Linting:**
- Use Ruff via `uv run ruff check .`; configuration is in `pyproject.toml`.
- Ruff excludes generated or externally sourced areas: `data`, `examples`, `.venv`, `build`, `dist`, and caches in `pyproject.toml`.
- Ruff ignores `E741`; avoid relying on that exception for new code unless matching nearby code is unavoidable.

## Import Organization

**Order:**
1. Future imports when needed, usually `from __future__ import annotations`, as in `src/sol_execbench/core/data/definition.py` and `tests/sol_execbench/test_e2e.py`.
2. Standard library imports: `json`, `subprocess`, `dataclasses`, `pathlib`, `typing`, `unittest.mock`.
3. Third-party imports: `pytest`, `pydantic`, `torch`, `click`, `rich`.
4. Local package imports from `sol_execbench...`, grouped by subsystem.

**Path Aliases:**
- No custom import aliases are configured in `pyproject.toml`; import package code through the installed package name `sol_execbench`.
- Tests import package modules directly, for example `from sol_execbench.core.data.solution import BuildSpec` in `tests/sol_execbench/core/data/test_solution.py`.
- Do not use relative imports in tests; keep tests aligned with installed-package behavior.

## Error Handling

**Patterns:**
- Public schema validation uses Pydantic validators that raise `ValueError` with actionable messages: `BuildSpec._reject_legacy_languages()` and `BuildSpec._validate_entry_point()` in `src/sol_execbench/core/data/solution.py`.
- File/path and subprocess boundaries raise domain-appropriate built-ins: `FileNotFoundError` for missing compiled artifacts in `src/sol_execbench/driver/problem_packager.py`, `RuntimeError` for evaluation/template execution failures in `src/sol_execbench/driver/templates/eval_driver.py`.
- CLI argument and problem resolution failures raise `click.ClickException` for user-facing messages in `src/sol_execbench/cli/main.py`.
- Security/integrity checks use a custom exception where callers need to distinguish reward-hack failures: `RewardHackDetected` in `src/sol_execbench/core/bench/reward_hack.py`.
- Preserve exception chaining when transforming parse/runtime errors: `raise ValueError(...) from exc` in `src/sol_execbench/core/baseline.py` and `src/sol_execbench/core/data/definition.py`.
- Best-effort cleanup should log and continue rather than masking the original workflow: `unlock_clocks()` in `src/sol_execbench/core/bench/clock_lock.py`.

## Logging

**Framework:** Python `logging` for library internals; Rich `Console` and `print` for CLI/template process output.

**Patterns:**
- Library modules that interact with external tools create `logger = logging.getLogger(__name__)`, as in `src/sol_execbench/core/bench/clock_lock.py`.
- Use `logger.warning()` for recoverable hardware/tooling failures and `logger.info()` for state transitions around ROCm clock locking in `src/sol_execbench/core/bench/clock_lock.py`.
- CLI modules use a module-level Rich console (`console = Console(stderr=True)`) and `console.print()` for tables, status messages, and diagnostics in `src/sol_execbench/cli/main.py`.
- Driver templates emit machine-readable JSON traces to stdout and push non-trace status/noise to stderr where possible; see `src/sol_execbench/driver/templates/eval_driver.py`.
- Scripts intended for direct operator use print progress and summaries directly, as in `scripts/run_dataset.py`.

## Comments

**When to Comment:**
- Use module docstrings to state subsystem purpose: `src/sol_execbench/core/data/json_utils.py`, `tests/sol_execbench/test_e2e.py`.
- Use comments to mark multi-step phases or security-sensitive boundaries: compile/evaluate phases in `tests/sol_execbench/test_e2e.py`, blocked `torch.utils.cpp_extension.load` behavior in `src/sol_execbench/driver/templates/eval_driver.py`.
- Avoid comments that restate a single obvious line; prefer descriptive helper names such as `_missing_safetensors_inputs()` in `tests/sol_execbench/test_e2e.py`.

**JSDoc/TSDoc:**
- Not applicable; this is a Python codebase.
- Use Python docstrings. Core data models use class and attribute docstrings because `BaseModelWithDocstrings` enables `ConfigDict(use_attribute_docstrings=True)` in `src/sol_execbench/core/data/base_model.py`.
- Longer function docstrings use NumPy-style sections (`Parameters`, `Returns`, `Raises`) in utility code such as `src/sol_execbench/core/data/json_utils.py`.

## Function Design

**Size:** Keep helpers focused around one operation. Schema/model validators in `src/sol_execbench/core/data/solution.py` each enforce one constraint; larger orchestration functions such as `cli()` in `src/sol_execbench/cli/main.py` are acceptable at command boundaries.

**Parameters:** Prefer typed `Path`, Pydantic models, and explicit collections. Examples include `_run_subprocess(cmd: list[str], cwd: Path)` in `tests/sol_execbench/test_e2e.py` and `load_json_file(model_cls: Type[T], path: Union[str, Path])` in `src/sol_execbench/core/data/json_utils.py`.

**Return Values:** Return typed domain objects or simple booleans for probes. Examples: `load_jsonl_file()` returns `list[T]` in `src/sol_execbench/core/data/json_utils.py`; `probe_clock_lock_available()` returns `bool` in `src/sol_execbench/core/bench/clock_lock.py`; `compare_trace_baselines()` returns `BaselineComparison` in `src/sol_execbench/core/baseline.py`.

## Module Design

**Exports:** Package-level exports are centralized in `src/sol_execbench/core/__init__.py`; new public core models/utilities should be added there only when they are part of the stable public surface.

**Barrel Files:** Minimal `__init__.py` files exist for packages such as `src/sol_execbench/cli/__init__.py`, `src/sol_execbench/driver/__init__.py`, and `src/sol_execbench/core/bench/__init__.py`. Keep barrel exports sparse.

**Model Modules:** Put Pydantic schema/domain types under `src/sol_execbench/core/data/`, keep validation close to the model, and use field/model validators for schema invariants. Examples: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/solution.py`.

**Driver Boundaries:** Keep subprocess packaging and generated scripts under `src/sol_execbench/driver/`; use `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py` as the boundary between Python objects and staged evaluation files.

**CLI Boundaries:** Keep user-facing command parsing and rendering under `src/sol_execbench/cli/`; call core/driver services rather than duplicating schema or benchmark logic.

---

*Convention analysis: 2026-05-22*
