# Coding Conventions

**Analysis Date:** 2026-06-01

## Naming Patterns

**Files:**
- Use `snake_case.py` for Python modules under `src/sol_execbench/`, `scripts/`, and `tests/`; examples include `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/driver/problem_packager.py`, and `tests/sol_execbench/test_cli_environment_snapshot.py`.
- Use `test_*.py` for tests, with deeper package mirrors for focused modules such as `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/core/data/test_solution.py`, and `tests/sol_execbench/driver/test_problem_packager.py`.
- Generated evaluation templates live as normal Python files in `src/sol_execbench/driver/templates/eval_driver.py` and `src/sol_execbench/driver/templates/build_ext.py`; keep template-compatible code importable when helper extraction is practical, as in `src/sol_execbench/core/bench/eval_runtime.py`.
- Public documentation uses uppercase names for broad guides such as `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `docs/TESTING.md`, and `docs/DEVELOPMENT.md`; topic docs under `docs/` use lowercase descriptive names such as `docs/rocm.md` and `docs/static_kernel_evidence.md`.

**Functions:**
- Use `snake_case` for functions and methods. Private helpers use a leading underscore, for example `_load_workloads` in `src/sol_execbench/cli/main.py`, `_inject_offload_arch_flags` in `src/sol_execbench/driver/problem_packager.py`, and `_run_eval_driver` in `tests/sol_execbench/driver/test_eval_driver.py`.
- Test functions must describe expected behavior, for example `test_legacy_cuda_nvidia_languages_rejected_with_guidance` in `tests/sol_execbench/core/data/test_solution.py` and `test_environment_snapshot_collection_failure_is_nonfatal` in `tests/sol_execbench/test_cli_environment_snapshot.py`.
- Use small helper functions in tests when fixture setup is repeated, such as `_make_spec` in `tests/sol_execbench/core/data/test_solution.py`, `_load_example` in `tests/examples/test_examples.py`, and `_make_packager` in `tests/sol_execbench/driver/test_problem_packager.py`.

**Variables:**
- Use `snake_case` for local variables and parameters. Use clear names for domain entities: `definition`, `workloads`, `solution`, `config`, `trace`, `sidecar`, and `staging_dir` appear throughout `src/sol_execbench/cli/main.py`, `src/sol_execbench/driver/problem_packager.py`, and `tests/sol_execbench/test_e2e.py`.
- Module constants use uppercase with leading underscores for private fixtures and category sets, for example `_CPP_LANGUAGES` in `src/sol_execbench/driver/problem_packager.py`, `_ROCM_DEVICE_NODES` in `tests/conftest.py`, and `_EXAMPLES` in `tests/examples/test_examples.py`.
- Environment variable names are uppercase constants near their use, such as `ENV_SNAPSHOT_ENABLE_ENV`, `ENV_SNAPSHOT_PATH_ENV`, `PROFILE_ROCPROFV3`, and `STATIC_EVIDENCE_AUTO` in `src/sol_execbench/cli/main.py`.

**Types:**
- Use `PascalCase` for classes, dataclasses, Pydantic models, and enums. Examples include `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py`, `EnvironmentSnapshot` in `src/sol_execbench/core/environment.py`, `SupportedLanguages` in `src/sol_execbench/core/data/solution.py`, and `Sample` in `tests/sol_execbench/test_e2e.py`.
- Enum classes derive from `str, Enum` when serialized into schemas; enum values are lowercase strings or stable public target names, as in `SupportedLanguages` and `SupportedHardware` in `src/sol_execbench/core/data/solution.py`.
- Prefer explicit built-in generics and union syntax (`list[str]`, `dict[str, Any]`, `Path | None`) with `from __future__ import annotations`, as used in `src/sol_execbench/core/environment.py`, `src/sol_execbench/driver/problem_packager.py`, and `tests/examples/test_examples.py`.
- Shared schema test helpers return concrete Pydantic types through constructors in `tests/sol_execbench_type_helpers.py`; use `make_definition`, `make_workload`, `make_solution`, `make_build_spec`, and `make_trace` for schema-heavy tests.

## Code Style

**Formatting:**
- Use Ruff formatting. `pyproject.toml` configures `[tool.ruff]`, and `.pre-commit-config.yaml` runs `ruff` with `--fix` plus `ruff-format`.
- Run `uv run ruff format .` for formatting, or `uv run --with ruff ruff format .` when Ruff is not already installed.
- Keep line wrapping and import layout compatible with Ruff defaults. Long assertions use parenthesized multi-line messages, as in `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.
- Generated data, downloaded datasets, `examples/`, build artifacts, caches, and virtual environments are excluded from Ruff in `pyproject.toml`; do not use examples as the formatting authority for package source.

**Linting:**
- Run `uv run ruff check .` for linting. CI runs the same command in `.github/workflows/code-quality.yml`.
- Ruff ignores only `E741` in `pyproject.toml`; avoid broad local ignores.
- Use Ty for static type checks with roots configured under `[tool.ty.src] include = ["src", "tests"]` in `pyproject.toml`. CI runs `uv run ty check`.
- The pre-commit configuration in `.pre-commit-config.yaml` also enforces DCO sign-off in commit messages; keep commits signed when contributing.

## Import Organization

**Order:**
1. `from __future__ import annotations` when needed, after SPDX/license header and module docstring conventions used by the file.
2. Standard library imports, alphabetized by Ruff, such as `json`, `subprocess`, `Path`, `dataclass`, and `typing`.
3. Third-party imports such as `click`, `pytest`, `pydantic`, `rich`, and `torch`.
4. Project imports from `sol_execbench` or relative package imports. Prefer explicit imports from the owning module when tests assert concrete behavior.

**Path Aliases:**
- No Python path alias system is configured. Source package imports use `sol_execbench...`, for example `from sol_execbench.driver.problem_packager import ProblemPackager` in `tests/sol_execbench/test_e2e.py`.
- Package-internal modules use relative imports where local ownership is clear, for example `from ..core import ...` in `src/sol_execbench/driver/problem_packager.py` and `from .data.base_model import BaseModelWithDocstrings` in `src/sol_execbench/core/environment.py`.
- Tests import shared helpers from `tests/sol_execbench_type_helpers.py` as the top-level module `sol_execbench_type_helpers`.

## Error Handling

**Patterns:**
- Raise `ValueError` for invalid data contracts, schema semantics, and report payloads. Examples include validators in `src/sol_execbench/core/data/solution.py`, matrix validation in `src/sol_execbench/core/matrix_diff.py`, and scoring input validation in `src/sol_execbench/core/scoring/amd_hardware_models.py`.
- Raise `click.ClickException` for CLI user errors in `src/sol_execbench/cli/main.py`, such as missing `definition.json`, missing `workload.jsonl`, or invalid command option combinations.
- Raise `RuntimeError` or `FileNotFoundError` for execution boundary failures, such as missing staged `definition.json` in `src/sol_execbench/core/bench/eval_runtime.py` and missing `benchmark_kernel.so` in `src/sol_execbench/driver/problem_packager.py`.
- Preserve exception causes with `raise ... from exc` when adding context around parser or conversion failures, as in `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/matrix_diff.py`, and `src/sol_execbench/core/scoring/amd_hardware_models.py`.
- Optional diagnostic evidence must be non-fatal: environment snapshots, profiling metadata, and static evidence sidecars catch broad exceptions at the CLI boundary in `src/sol_execbench/cli/main.py` and print warnings instead of changing benchmark correctness status.
- Subprocess-facing helpers use bounded timeouts and injectable runners where behavior needs tests. Use `runner` parameters as in `src/sol_execbench/core/environment.py` and `src/sol_execbench/core/bench/rocm_profiler.py`.

## Logging

**Framework:** Python `logging` for library internals; Rich `Console` and `print` for CLI/script output.

**Patterns:**
- Use a module logger for library warnings and info when not directly producing CLI output. `src/sol_execbench/core/bench/clock_lock.py` defines `logger = logging.getLogger(__name__)` and logs clock-lock warnings.
- Use `Console(stderr=True)` in the Click CLI for human-facing status, tables, warnings, and runtime logs, as in `src/sol_execbench/cli/main.py`.
- Use JSON `print` only for machine-readable CLI/report output, such as trace JSONL emission in `src/sol_execbench/cli/main.py` and command handlers in `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, and `src/sol_execbench/core/runtime_evidence.py`.
- Scripts under `scripts/` use `print` for batch progress and summaries, for example `scripts/run_dataset.py` and `scripts/download_solexecbench.py`.

## Comments

**When to Comment:**
- Include module docstrings describing purpose and runtime boundaries, especially for public modules and generated templates; examples are `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/core/environment.py`, and `tests/sol_execbench/driver/test_eval_driver.py`.
- Use short comments to separate complex test sections or runtime phases, as in `tests/sol_execbench/test_e2e.py`, `tests/examples/test_examples.py`, and `tests/sol_execbench/driver/test_eval_driver.py`.
- Keep comments tied to non-obvious behavior: ROCm migration guidance, reward-hack defenses, clock-lock semantics, evidence authority boundaries, and subprocess staging.

**JSDoc/TSDoc:**
- Not applicable. This is a Python codebase.
- Python docstrings are used for public functions, model classes, validators, and fixtures. Pydantic schema models use attribute docstrings through `BaseModelWithDocstrings` in `src/sol_execbench/core/data/base_model.py`.

## Function Design

**Size:** Keep new helpers focused and move reusable boundary logic out of large orchestrators when a coherent extraction exists. Current helper modules include `src/sol_execbench/core/dataset/run_state.py`, `src/sol_execbench/core/dataset/evidence_refs.py`, `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/core/scoring/amd_bound_classification.py`, and `src/sol_execbench/core/bench/static_kernel_status.py`.

**Parameters:** Prefer explicit typed parameters over ambient state. For filesystem and subprocess code, pass `Path`, command lists, timeout values, and injectable callables (`runner`, `which`, `now`, `collector`) as seen in `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and `src/sol_execbench/cli/main.py`.

**Return Values:** Return typed models, dataclasses, or simple tuples/lists with documented meaning. Examples include `ProblemPackager.compile() -> tuple[list[str], str]` in `src/sol_execbench/driver/problem_packager.py`, `load_staged_problem() -> tuple[dict[str, Any], list[dict[str, Any]]]` in `src/sol_execbench/core/bench/eval_runtime.py`, and Pydantic sidecar/report models under `src/sol_execbench/core/`.

## Module Design

**Exports:** Package `__init__.py` files expose stable public imports. `src/sol_execbench/core/__init__.py` and `src/sol_execbench/driver/__init__.py` are used by callers and tests; add exports there only for stable project APIs.

**Barrel Files:** Barrel-style exports are used sparingly for package convenience. Keep implementation in focused modules (`src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/driver/problem_packager.py`) and import through package barrels only where existing code already does so.

---

*Convention analysis: 2026-06-01*
