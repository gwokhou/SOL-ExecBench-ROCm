# Coding Conventions

**Analysis Date:** 2026-06-04

## Naming Patterns

**Files:**
- Use lowercase `snake_case.py` for Python modules under `src/sol_execbench/`, `tests/sol_execbench/`, and `scripts/`; examples include `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `tests/sol_execbench/test_rocm_profiler.py`, and `scripts/run_dataset.py`.
- Keep implementation modules near their domain package: data schemas in `src/sol_execbench/core/data/`, benchmark runtime code in `src/sol_execbench/core/bench/`, dataset logic in `src/sol_execbench/core/dataset/`, scoring logic in `src/sol_execbench/core/scoring/`, and CLI code in `src/sol_execbench/cli/`.
- Test modules use `test_*.py` names and mirror either package paths (`tests/sol_execbench/core/data/test_solution.py`) or feature workflows (`tests/sol_execbench/test_run_docker_dependency_preflight.py`).
- Generated driver templates live under `src/sol_execbench/driver/templates/`; treat `src/sol_execbench/driver/templates/eval_driver.py` and `src/sol_execbench/driver/templates/build_ext.py` as executable source templates that need syntax and subprocess tests.

**Functions:**
- Use `snake_case` for public helpers and test functions: `collect_rocprofv3_profile` in `src/sol_execbench/core/bench/rocm_profiler.py`, `build_toolchain_routing_report` in `src/sol_execbench/core/toolchain.py`, and `test_dangerous_native_compile_options_rejected` in `tests/sol_execbench/core/data/test_solution.py`.
- Prefix internal helpers with `_`, especially file, parsing, and validation helpers: `_validate_compile_flag` in `src/sol_execbench/core/data/solution.py`, `_run_eval_driver_process` in `tests/sol_execbench/driver/test_eval_driver.py`, and `_matching_default_dependency_env` in `tests/sol_execbench/test_run_docker_dependency_preflight.py`.
- Use verb-led names for operations that have effects or construct artifacts: `write_migration_manifest` in `src/sol_execbench/core/dataset/migration.py`, `discover_rocprofv3_artifacts` in `src/sol_execbench/core/bench/rocm_profiler.py`, and `save_cli_log` in `src/sol_execbench/core/dataset/runner.py`.

**Variables:**
- Use `snake_case` for local variables and parameters.
- Use uppercase constants for schema versions, environment variable names, path allowlists, and shared static sets: `ROCPROFV3_PROFILE_SCHEMA_VERSION` in `src/sol_execbench/core/bench/rocm_profiler.py`, `ENV_SNAPSHOT_ENABLE_ENV` in `src/sol_execbench/cli/main.py`, `_PATH_INJECTION_PREFIXES` in `src/sol_execbench/core/data/solution.py`, and `NATIVE_ROCM_LANGUAGES` in `src/sol_execbench/core/bench/eval_runtime.py`.
- Use precise typed aliases for callback boundaries: `ProfilerRunner` and `ProfileRunner` in `src/sol_execbench/core/bench/rocm_profiler.py`, `PathExists` in `tests/conftest.py`, and `JsonDict` in `tests/sol_execbench_type_helpers.py`.

**Types:**
- Use `PascalCase` for classes, Pydantic models, dataclasses, and enums: `SourceFile`, `CompileOptions`, and `BuildSpec` in `src/sol_execbench/core/data/solution.py`; `Rocprofv3ProfileRequest` and `Rocprofv3ProfileResult` in `src/sol_execbench/core/bench/rocm_profiler.py`.
- Enum classes use `PascalCase`; enum members use uppercase names with string values, as in `SupportedLanguages.HIP_CPP` and `SupportedHardware.GFX1200` in `src/sol_execbench/core/data/solution.py`.
- Prefer frozen dataclasses for immutable evidence and result objects: `Rocprofv3TimingRow`, `Rocprofv3TimingEvidence`, and `DefaultTimingSelection` in `src/sol_execbench/core/bench/rocm_profiler.py`.
- Pydantic models share `BaseModelWithDocstrings` from `src/sol_execbench/core/data/base_model.py` so attribute docstrings flow into JSON schema.

## Code Style

**Formatting:**
- Use Ruff formatting. Run `uv run --with ruff ruff format .`.
- Ruff is configured in `pyproject.toml` and force-excludes generated or downloaded paths: `.git`, `.venv`, `__pycache__`, `build`, `dist`, `*.egg-info`, `.ruff_cache`, `data`, and `examples`.
- Python requires `>=3.12,<3.14` in `pyproject.toml`; use native Python 3.12 type syntax such as `list[str]`, `Path | None`, and `tuple[str, ...]`.
- Keep line wrapping consistent with Ruff. Long assertions may use parenthesized message expressions, as in `tests/sol_execbench/driver/test_eval_driver.py`.

**Linting:**
- Use Ruff linting. Run `uv run --with ruff ruff check .`.
- Ruff ignores only `E741` in `pyproject.toml`; avoid introducing broad local disables.
- Type checking uses `ty` over `src` and `tests` via `[tool.ty.src]` in `pyproject.toml`; keep test helpers typed enough for `ty`, using explicit casts in `tests/sol_execbench_type_helpers.py` when needed.
- Development dependencies live in the `dev` dependency group in `pyproject.toml`; use `uv sync --all-groups` or targeted `uv run` commands.

## Import Organization

**Order:**
1. `from __future__ import annotations` after license/module docstring when used, as in `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and many tests.
2. Standard library imports, grouped alphabetically enough for readability: `json`, `os`, `subprocess`, `sys`, `Path`, and typing imports in `src/sol_execbench/cli/main.py`.
3. Third-party imports such as `click`, `rich`, `pytest`, `torch`, and `pydantic`.
4. Local package imports from `sol_execbench` or relative imports inside `src/sol_execbench/`; CLI code uses relative imports such as `from ..core import Definition` in `src/sol_execbench/cli/main.py`.

**Path Aliases:**
- No project path alias system is configured. Runtime package imports use `sol_execbench.*`.
- Tests import source through the installed/editable package and use helper module `tests/sol_execbench_type_helpers.py`.
- Docker/preflight tests set `PYTHONPATH` explicitly when invoking scripts, as in `_run_docker_dependency_preflight` in `tests/sol_execbench/test_run_docker_dependency_preflight.py`.

## Error Handling

**Patterns:**
- Raise `ValueError` for invalid data, unsupported schema values, and unsafe user-controlled paths or flags. Examples: `_validate_compile_flag` in `src/sol_execbench/core/data/solution.py`, `validate_sidecar_identifier` in `src/sol_execbench/core/dataset/evidence_refs.py`, and Docker target validation in `src/sol_execbench/core/docker_matrix.py`.
- Raise `RuntimeError` for runtime failures, missing generated artifacts, or staged execution failures. Examples: `load_staged_problem` and `load_reference_function` in `src/sol_execbench/core/bench/eval_runtime.py`.
- Raise `click.ClickException` for user-facing CLI validation in `src/sol_execbench/cli/main.py`, such as missing `definition.json`, unsupported output modes, or incomplete CLI arguments.
- Preserve failure causes with `raise ... from exc` where the original exception helps diagnostics, as in JSON parsing and module execution paths in `src/sol_execbench/core/baseline.py` and `src/sol_execbench/core/bench/eval_runtime.py`.
- Return explicit status metadata rather than throwing for optional diagnostics. `collect_rocprofv3_profile` in `src/sol_execbench/core/bench/rocm_profiler.py` returns `status`, `skipped_reason`, `failed_reason`, tails, return codes, and artifact metadata.
- Keep no-trace and profiler diagnostics bounded. `_diagnostic_tail` and `_write_no_trace_diagnostics_sidecar` in `src/sol_execbench/cli/main.py` cap stdout/stderr evidence at `_DIAGNOSTIC_TAIL_LIMIT`.

## Logging

**Framework:** `logging` for reusable library behavior; Rich console output for CLI UX.

**Patterns:**
- Use module loggers in library code that performs environment or hardware operations, e.g. `logger = logging.getLogger(__name__)` in `src/sol_execbench/core/bench/clock_lock.py`.
- Use `console.print` only in CLI modules such as `src/sol_execbench/cli/main.py`; do not add Rich output to core library modules.
- For subprocess diagnostics, persist structured JSON sidecars or include stderr/stdout tails in result objects instead of logging unbounded output. Follow `Rocprofv3ProfileResult.to_dict` in `src/sol_execbench/core/bench/rocm_profiler.py` and `_write_no_trace_diagnostics_sidecar` in `src/sol_execbench/cli/main.py`.

## Comments

**When to Comment:**
- Use docstrings for public modules, classes, dataclasses, validators, fixtures, and non-obvious helpers. Examples: `SourceFile` in `src/sol_execbench/core/data/solution.py`, `Rocprofv3TimingEvidence` in `src/sol_execbench/core/bench/rocm_profiler.py`, and `tmp_cache_dir` in `tests/conftest.py`.
- Use short inline comments to explain non-obvious boundary decisions, such as CPU timing opt-in in `src/sol_execbench/core/bench/eval_runtime.py`, offload architecture injection in `src/sol_execbench/driver/problem_packager.py`, and timing marker skip policy in `tests/conftest.py`.
- Keep section comments in long integration tests when they improve scanning, as in `tests/sol_execbench/driver/test_eval_driver.py`.

**JSDoc/TSDoc:**
- Not applicable. This repository is Python-only for source and tests.

## Function Design

**Size:** Keep core helpers small and single-purpose. Large orchestration functions are acceptable at CLI/script boundaries (`src/sol_execbench/cli/main.py`, `scripts/run_dataset.py`) but new core logic should be factored into testable helpers.

**Parameters:** Use typed parameters and keyword-only options for configurable behavior. Examples include `measure_latency(..., *, warmup, rep, time_fn=None)` in `src/sol_execbench/core/bench/eval_runtime.py` and `build_rocprofv3_command(..., *, output_directory, output_file, executable, include_hip_runtime)` in `src/sol_execbench/core/bench/rocm_profiler.py`.

**Return Values:** Prefer typed dataclasses or Pydantic models for structured outputs. Return `subprocess.CompletedProcess[str]` from injected runners and subprocess helpers, as in `tests/sol_execbench/test_run_docker_dependency_preflight.py` and `src/sol_execbench/core/bench/rocm_profiler.py`.

## Module Design

**Exports:** Package `__init__.py` files re-export public domain objects for stable imports, especially `src/sol_execbench/core/__init__.py` and `src/sol_execbench/driver/__init__.py`. Add public exports deliberately.

**Barrel Files:** Barrel-style package exports are used sparingly. Prefer direct module imports in tests when validating module internals, such as `from sol_execbench.core.data.solution import CompileOptions`.

## Subprocess Practices

- Build commands as `list[str]` with no shell interpolation. Examples: `ProblemPackager.compile` and `ProblemPackager.execute` in `src/sol_execbench/driver/problem_packager.py`, `build_rocprofv3_command` in `src/sol_execbench/core/bench/rocm_profiler.py`, and `_run_docker_dependency_preflight` in `tests/sol_execbench/test_run_docker_dependency_preflight.py`.
- Always provide `timeout=` for live subprocess calls. Examples: CLI compile/evaluation calls in `src/sol_execbench/cli/main.py`, docker dependency tests in `tests/docker/dependencies/test_rocm_runtime.py`, and E2E calls in `tests/sol_execbench/test_e2e.py`.
- Use `capture_output=True` and `text=True` for subprocesses whose diagnostics or JSON output are asserted.
- Inject runner callables for unit-testable command execution boundaries. Follow `ProfilerRunner` and `ProfileRunner` in `src/sol_execbench/core/bench/rocm_profiler.py`.
- Do not let diagnostic subprocess noise corrupt canonical trace JSONL. `src/sol_execbench/driver/templates/eval_driver.py` and `tests/sol_execbench/driver/test_eval_driver.py` enforce stdout JSONL separation from stderr user noise.

## Security Practices

- Reject absolute paths and `..` traversal for user-provided source paths. Follow `SourceFile._validate_source_path` in `src/sol_execbench/core/data/solution.py`.
- Reject native compile flags that can reference host paths, response files, sysroots, plugins, or runtime linker paths. Follow `_validate_compile_flag` in `src/sol_execbench/core/data/solution.py`.
- Only allow explicitly safe ROCm include/library flags such as `-I/opt/rocm/include` and `-L/opt/rocm/lib` in `src/sol_execbench/core/data/solution.py`.
- Block dynamic PyTorch C++ extension builds inside GPU evaluation. Use `block_cpp_extension_load` in `src/sol_execbench/core/bench/eval_runtime.py` rather than allowing user code to compile on the GPU server.
- Detect reward-hack behavior around monkey patches, thread injection, lazy outputs, and tampered scoring helpers in `src/sol_execbench/core/bench/reward_hack.py`; add tests under `tests/sol_execbench/core/bench/test_reward_hack.py` or `tests/sol_execbench/driver/test_eval_driver.py`.
- Treat profiler outputs and environment snapshots as diagnostic-only unless code explicitly marks them authoritative. `Rocprofv3ProfileResult.to_dict` in `src/sol_execbench/core/bench/rocm_profiler.py` sets `diagnostic_only` and `score_authority`.
- Never read, print, or persist secret environment values in new diagnostics. Existing environment reporting code lives in `src/sol_execbench/core/environment.py`; keep additions schema-oriented and bounded.

---

*Convention analysis: 2026-06-04*
