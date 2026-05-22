# Coding Conventions

**Analysis Date:** 2026-05-22

## Naming Patterns

**Files:**
- Use lowercase `snake_case.py` for Python modules: `src/sol_execbench/core/bench/timing_policy.py`, `src/sol_execbench/core/scoring/baseline_artifact.py`.
- Test files use `test_*.py` and mirror source areas where possible: `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/driver/test_problem_packager.py`.
- Template scripts in `src/sol_execbench/driver/templates/` are normal Python files named for the staged action: `build_ext.py`, `eval_driver.py`.
- Example kernels and references use fixed names inside each example directory: `examples/triton/rmsnorm/kernel.py`, `examples/triton/rmsnorm/reference.py`.

**Functions:**
- Use `snake_case` for public and private functions: `time_runnable()` in `src/sol_execbench/core/bench/timing.py`, `_get_local_gfx()` in `src/sol_execbench/driver/problem_packager.py`.
- Prefix helper functions with `_` when module-private: `_load_solution()` in `src/sol_execbench/cli/main.py`, `_inject_offload_arch_flags()` in `src/sol_execbench/driver/problem_packager.py`.
- Test helper factories use `_make_*`, `_load_*`, or `_run_*`: `_make_spec()` in `tests/sol_execbench/core/data/test_solution.py`, `_exec_build_ext()` in `tests/sol_execbench/driver/test_build_ext.py`.
- Pydantic validators use explicit `_validate_*` or `_reject_*` names and decorators: `_validate_source_path()` and `_reject_legacy_languages()` in `src/sol_execbench/core/data/solution.py`.

**Variables:**
- Use `snake_case` for locals and fields: `compile_options`, `hip_cflags`, `output_dir`, `target_hardware`.
- Constants use uppercase module-level names: `_CPP_LANGUAGES` in `src/sol_execbench/driver/problem_packager.py`, `ROCPROFV3_EXECUTABLE` in `src/sol_execbench/core/bench/rocm_profiler.py`.
- Test data constants use leading underscores when private to a test module: `_DEFINITION_DICT`, `_WORKLOAD_DICTS`, `_HIP_SOLUTION_DICT` in `tests/sol_execbench/driver/test_problem_packager.py`.
- Prefer concrete names over abbreviations except well-known ROCm terms: `gfx_arch`, `rocprofv3_available`, `kernel_duration_ms`.

**Types:**
- Use `PascalCase` for classes, dataclasses, Pydantic models, and enums: `ProblemPackager`, `Rocprofv3TimingEvidence`, `SupportedLanguages`, `Definition`.
- Enum classes use `PascalCase`; enum members are uppercase for symbolic options: `SupportedLanguages.HIP_CPP`, `SupportedHardware.GFX1200` in `src/sol_execbench/core/data/solution.py`.
- Type aliases use `PascalCase` when exported: `NonEmptyString`, `NonNegativeInt` in `src/sol_execbench/core/data/base_model.py`.
- Use built-in generic syntax and unions: `list[Workload]`, `dict[str, object]`, `Path | None`, visible throughout `src/sol_execbench/driver/problem_packager.py` and `src/sol_execbench/core/bench/rocm_profiler.py`.

## Code Style

**Formatting:**
- Use Ruff formatting via `uv run ruff format .`; Ruff excludes `data` and `examples` in `pyproject.toml`.
- Target Python is `>=3.12,<3.14` in `pyproject.toml`; use Python 3.12 syntax, including `|` unions and generic built-ins.
- Keep SPDX copyright and Apache-2.0 license headers at the top of source and test modules. Most files under `src/sol_execbench/` and `tests/sol_execbench/` include this header.
- Add a concise module docstring after the license header for non-trivial modules: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `tests/sol_execbench/test_e2e.py`.

**Linting:**
- Use Ruff via `uv run ruff check .`.
- Ruff ignores `E741` in `pyproject.toml`; avoid ambiguous one-letter names anyway except short loop variables in local comprehensions.
- Generated data, downloaded datasets, and examples are excluded from Ruff checks by `pyproject.toml`; keep production package style stricter than example code.

## Import Organization

**Order:**
1. `from __future__ import annotations` when needed, immediately after the module docstring or license/docstring block: `src/sol_execbench/cli/main.py`, `tests/examples/test_examples.py`.
2. Standard library imports: `json`, `subprocess`, `Path`, `dataclass`, `Sequence`.
3. Third-party imports: `click`, `pytest`, `torch`, `pydantic`, `rich`.
4. Local package imports: `from sol_execbench.core...` in tests, relative imports inside package modules such as `from ..core import ...` in `src/sol_execbench/cli/main.py`.
5. `TYPE_CHECKING` imports stay guarded to avoid runtime dependency work: `src/sol_execbench/core/data/definition.py`.

**Path Aliases:**
- No custom import alias configuration is detected in `pyproject.toml`.
- Use package-root imports in tests: `from sol_execbench.core import Definition`.
- Use relative imports inside `src/sol_execbench/` when importing sibling package layers: `from .base_model import ...`, `from ..core import ...`.

## Error Handling

**Patterns:**
- Use `ValueError` for schema, validation, parsing, and unsupported-option failures: `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/bench/timing.py`.
- Chain exceptions when converting parse errors into domain messages: `raise ValueError(...) from exc` in `src/sol_execbench/core/baseline.py` and `src/sol_execbench/core/data/definition.py`.
- Use `RuntimeError` for evaluation/runtime failures and output normalization failures: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/driver/templates/eval_driver.py`.
- Use `FileNotFoundError` for required generated artifacts: `benchmark_kernel.so` checks in `src/sol_execbench/driver/problem_packager.py` and `src/sol_execbench/driver/templates/build_ext.py`.
- Use `click.ClickException` for CLI-facing invalid arguments or missing problem files: `src/sol_execbench/cli/main.py`.
- Catch broad exceptions only at process/tooling boundaries where failure is intentionally converted to fallback behavior: ROCm tool detection in `src/sol_execbench/driver/problem_packager.py`, reward-hack isolation in `src/sol_execbench/core/bench/reward_hack.py`, diagnostics probing in `src/sol_execbench/core/diagnostics.py`.
- Preserve ROCm migration guidance in validation messages. For example, CUDA/NVIDIA schema values are rejected with replacement guidance in `src/sol_execbench/core/data/solution.py`.

## Logging

**Framework:** `logging` for library internals; Rich/Click output for CLI; `print()` only in staged driver/template subprocesses.

**Patterns:**
- Library modules that perform system operations log warnings instead of printing: `src/sol_execbench/core/bench/clock_lock.py`.
- CLI output uses a module-level Rich `Console(stderr=True)` and `console.print()` for status, tables, and subprocess logs: `src/sol_execbench/cli/main.py`.
- Staged subprocess templates emit machine-readable trace JSON on stdout and operational messages on stderr or explicit print paths: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`.
- Do not add ad hoc `print()` calls to reusable package modules under `src/sol_execbench/core/`; return structured data or log instead.

## Comments

**When to Comment:**
- Keep comments for ROCm compatibility context, subprocess phase boundaries, or subtle timing semantics: `src/sol_execbench/core/bench/timing.py`, `tests/examples/test_examples.py`.
- Use short comments before non-obvious validation or staging behavior, such as source resolution in `src/sol_execbench/cli/main.py` and offload architecture injection in `src/sol_execbench/driver/problem_packager.py`.
- Avoid comments that restate the line of code. Prefer docstrings for public behavior and tests for expected outcomes.

**JSDoc/TSDoc:**
- Not applicable; this is a Python repository.
- Use Python docstrings on public functions, dataclasses, Pydantic models, validators, and test classes. Model fields often use attribute docstrings for JSON schema generation via `BaseModelWithDocstrings` in `src/sol_execbench/core/data/base_model.py`.

## Function Design

**Size:** Keep pure helpers small and single-purpose. Larger orchestration functions are acceptable at subprocess or CLI boundaries, such as `cli()` in `src/sol_execbench/cli/main.py` and staged evaluation code in `src/sol_execbench/driver/templates/eval_driver.py`.

**Parameters:** Use typed parameters and concrete containers. Prefer dependency injection for external commands or profiling runners, as in `collect_rocprofv3_timing(..., runner=runner)` in `src/sol_execbench/core/bench/rocm_profiler.py`.

**Return Values:** Return structured domain objects or simple tuples/lists. Examples: `ProblemPackager.compile()` returns `(command, artifact_path)` in `src/sol_execbench/driver/problem_packager.py`; profiler helpers return dataclasses with `to_dict()` methods in `src/sol_execbench/core/bench/rocm_profiler.py`.

## Module Design

**Exports:** Use package `__init__.py` files to expose public core types for ergonomic imports. Tests import `Definition`, `Solution`, `Trace`, and `BenchmarkConfig` from `sol_execbench.core`.

**Barrel Files:** Barrel-style exports are used in `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/data/__init__.py`, `src/sol_execbench/core/scoring/__init__.py`, and `src/sol_execbench/driver/__init__.py`. Add new public APIs there only when callers should import them from the package layer.

---

*Convention analysis: 2026-05-22*
