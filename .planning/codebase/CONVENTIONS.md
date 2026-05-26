# Coding Conventions

**Analysis Date:** 2026-05-26

## Naming Patterns

**Files:**
- Use lowercase `snake_case.py` modules for source and tests, as in `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/bench/clock_lock.py`, and `tests/sol_execbench/core/data/test_solution.py`.
- Name tests `test_*.py` and keep them near their target source area, such as `tests/sol_execbench/core/bench/test_clock_lock.py` for `src/sol_execbench/core/bench/clock_lock.py`.
- Keep runnable samples and malicious fixtures in explicit sample directories, such as `tests/sol_execbench/samples/evil_monkey_patch/kernel.py` and `tests/samples/rmsnorm/solution_hack_lazy_output.json`.

**Functions:**
- Use `snake_case` for public and private functions, for example `compare_trace_baselines()` in `src/sol_execbench/core/baseline.py` and `_validate_source_path()` in `src/sol_execbench/core/data/solution.py`.
- Prefix internal helpers with `_`, as in `_latency_ms()` and `_classify()` in `src/sol_execbench/core/baseline.py`.
- Use verb-led names for behavior, validation, and conversion: `load_trace_jsonl()` in `src/sol_execbench/core/baseline.py`, `format_baseline_comparison()` in `src/sol_execbench/core/baseline.py`, and `probe_clock_lock_available()` in `src/sol_execbench/core/bench/clock_lock.py`.

**Variables:**
- Use descriptive `snake_case` names for local values and parameters, such as `candidate_traces`, `baseline_traces`, and `output_file` in `src/sol_execbench/cli/baseline.py`.
- Use uppercase constants for module-level values, such as `VERIFY_DELAY_S` in `src/sol_execbench/core/bench/clock_lock.py`, `_ROCM_DEVICE_NODES` in `tests/conftest.py`, and `_EXAMPLES_DIR` in `tests/examples/test_examples.py`.
- Use leading underscores for module-private fixtures and data tables, such as `_DEFINITION_DICT` and `_WORKLOAD_DICTS` in `tests/sol_execbench/driver/test_problem_packager.py`.

**Types:**
- Use `PascalCase` for classes, dataclasses, Pydantic models, and enum classes, as in `Solution`, `BuildSpec`, and `SupportedLanguages` in `src/sol_execbench/core/data/solution.py`.
- Use uppercase enum members with string values, such as `SupportedLanguages.HIP_CPP = "hip_cpp"` in `src/sol_execbench/core/data/solution.py` and `EvaluationStatus.PASSED` in `src/sol_execbench/core/data/trace.py`.
- Use `dataclass(frozen=True)` for immutable value objects that are not schema models, such as `BaselineResult` in `src/sol_execbench/core/baseline.py` and `StageDiagnostic` in `src/sol_execbench/core/diagnostics.py`.
- Use `BaseModelWithDocstrings` for public Pydantic schemas so attribute docstrings appear in JSON schema, as defined in `src/sol_execbench/core/data/base_model.py`.

## Code Style

**Formatting:**
- Use Ruff for linting and formatting; configuration is in `pyproject.toml` under `[tool.ruff]`.
- Keep Python syntax compatible with `requires-python = ">=3.12,<3.14"` in `pyproject.toml`.
- Keep generated code, downloaded datasets, and examples out of Ruff checks; `pyproject.toml` excludes `data` and `examples`.
- Use modern type syntax such as `Path | None`, `list[Trace]`, and `dict[str, Any]`, as seen in `src/sol_execbench/core/baseline.py` and `src/sol_execbench/cli/baseline.py`.
- Preserve SPDX license headers in source and test files that already use them, such as `src/sol_execbench/core/data/solution.py` and `tests/sol_execbench/core/bench/test_clock_lock.py`.

**Linting:**
- Run `uv run ruff check .` for lint checks; CI runs this in `.github/workflows/code-quality.yml`.
- Run `uv run ruff format .` before broad formatting changes; development docs list this in `docs/DEVELOPMENT.md`.
- Ruff ignores `E741` in `pyproject.toml`, so avoid relying on ambiguous single-letter names even though that rule is disabled.
- Run `uv run ty check` for type checking over `src` and `tests`; source roots are configured in `pyproject.toml` under `[tool.ty.src]`.

## Import Organization

**Order:**
1. `from __future__ import annotations` when needed, as in `src/sol_execbench/core/baseline.py` and `tests/examples/test_examples.py`.
2. Standard library imports, grouped alphabetically by module style, such as `json`, `dataclasses`, `pathlib`, and `subprocess`.
3. Third-party imports, such as `click`, `pytest`, `pydantic`, and `torch`.
4. Package imports from `sol_execbench` or relative imports inside `src/sol_execbench/`.

**Path Aliases:**
- No custom import path aliases are configured in `pyproject.toml`; import package code through `sol_execbench...` in tests, as in `tests/sol_execbench/core/data/test_solution.py`.
- Source modules use relative imports within package subtrees, such as `from .data.trace import EvaluationStatus, Trace` in `src/sol_execbench/core/baseline.py`.
- Shared test builders live in `tests/sol_execbench_type_helpers.py`; use `make_definition()`, `make_solution()`, and related helpers instead of constructing typed objects through loose casts in each test.

## Error Handling

**Patterns:**
- Raise `ValueError` for invalid schema and domain input, with messages that include the offending field or value; examples are in `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/definition.py`, and `src/sol_execbench/core/scoring/amd_hardware_models.py`.
- Preserve exception context with `raise ... from exc` when converting parse or subprocess errors, as in `load_trace_jsonl()` in `src/sol_execbench/core/baseline.py` and `probe_toolchain_tool()` in `src/sol_execbench/core/toolchain.py`.
- Use `click.ClickException` for CLI validation errors, as in `src/sol_execbench/cli/main.py`.
- Treat optional evidence collection as nonfatal in CLI code; catch exceptions and print yellow skip messages in helpers such as `_write_environment_snapshot_sidecar()` and `_write_static_evidence_sidecar()` in `src/sol_execbench/cli/main.py`.
- Use explicit custom exception types for benchmark integrity failures, such as `RewardHackDetected` in `src/sol_execbench/core/bench/reward_hack.py` and `SolExecBenchError` in `src/sol_execbench/core/diagnostics.py`.
- For external commands, return structured status where possible instead of throwing directly; see `probe_tool()` in `src/sol_execbench/core/environment.py` and `probe_toolchain_tool()` in `src/sol_execbench/core/toolchain.py`.

## Logging

**Framework:** Python `logging` for library code; Click/Rich output for CLIs.

**Patterns:**
- Define `logger = logging.getLogger(__name__)` at module scope when library code needs diagnostics, as in `src/sol_execbench/core/bench/clock_lock.py`.
- Log recoverable hardware/tooling failures with `logger.warning()` and return `False` or a diagnostic object, as in `verify_clocks()` and `unlock_clocks()` in `src/sol_execbench/core/bench/clock_lock.py`.
- Use `console.print()` from Rich for the main CLI report path in `src/sol_execbench/cli/main.py`.
- Use `click.echo()` for simple CLI output in `src/sol_execbench/cli/baseline.py`.
- Do not print from low-level schema/model helpers such as `src/sol_execbench/core/data/solution.py`; raise typed errors or return data.

## Comments

**When to Comment:**
- Prefer docstrings for public modules, classes, and functions, as in `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/baseline.py`, and `tests/examples/test_examples.py`.
- Use short inline comments only when they explain non-obvious behavior, such as the immutable hash cache in `src/sol_execbench/core/data/solution.py` or the timing marker selection behavior in `tests/conftest.py`.
- Keep section comments in long test files to separate fixture setup, helpers, and parametrized cases, as in `tests/examples/test_examples.py` and `tests/sol_execbench/driver/test_problem_packager.py`.

**JSDoc/TSDoc:**
- Not applicable; this is a Python repository.

## Function Design

**Size:** Keep ordinary helpers small and single-purpose, such as `_format_latency()` in `src/sol_execbench/core/baseline.py`. Larger orchestration functions are acceptable at CLI, staging, and generated-driver boundaries, such as `_evaluate_cli()` in `src/sol_execbench/cli/main.py` and `ProblemPackager` methods in `src/sol_execbench/driver/problem_packager.py`.

**Parameters:** Use typed parameters and keyword-only options for behavioral knobs, as in `compare_trace_baselines()` in `src/sol_execbench/core/baseline.py` and `build_toolchain_routing_report()` in `src/sol_execbench/core/toolchain.py`.

**Return Values:** Return typed dataclasses, Pydantic models, or JSON-serializable dictionaries at boundaries. Examples include `BaselineComparison` from `compare_trace_baselines()` in `src/sol_execbench/core/baseline.py`, `EvaluatorContract.to_dict()` in `src/sol_execbench/core/data/contract.py`, and profile metadata `to_dict()` methods in `src/sol_execbench/core/bench/rocm_profiler.py`.

## Module Design

**Exports:** Use explicit `__all__` where a module exposes a small public API, as in `src/sol_execbench/core/baseline.py`. Package initializers such as `src/sol_execbench/core/__init__.py` re-export common core types for tests and callers.

**Barrel Files:** Use package `__init__.py` files sparingly as import conveniences; tests often import directly from implementation modules for focused behavior, such as `tests/sol_execbench/core/bench/test_clock_lock.py`.

**Pydantic Models:** Put public schemas under `src/sol_execbench/core/data/` and use Pydantic v2 validators with `@field_validator` and `@model_validator`, as in `src/sol_execbench/core/data/solution.py` and `src/sol_execbench/core/data/trace.py`.

**CLI Modules:** Keep Click command definitions under `src/sol_execbench/cli/`; `src/sol_execbench/cli/main.py` owns the main `sol-execbench` command and `src/sol_execbench/cli/baseline.py` owns `sol-execbench-baseline`.

**Testing Support:** Put reusable test-only factories in `tests/sol_execbench_type_helpers.py`, and keep domain fixtures near their tests, such as `tests/sol_execbench/solar_derivation_fixtures.py`.

---

*Convention analysis: 2026-05-26*
