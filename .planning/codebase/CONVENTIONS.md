# Coding Conventions

**Analysis Date:** 2026-06-04

## Naming Patterns

**Files:**
- Python package modules use `snake_case.py` under `src/sol_execbench/`, for
  example `src/sol_execbench/core/scoring/amd_bound_sanity.py` and
  `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- Test files use `test_*.py` in a separate `tests/` tree that mirrors package
  areas where useful, for example
  `tests/sol_execbench/core/bench/test_eval_runtime.py` and
  `tests/sol_execbench/driver/test_problem_packager.py`.
- Example problem assets use stable benchmark filenames:
  `definition.json`, `workload.jsonl`, `reference.py`, `kernel.py`,
  `kernel.hip`, `main.cpp`, and `solution_<backend>.json`.
- Generated data, downloaded benchmark assets, examples, and build/cache output
  are not part of normal formatting or linting scope.

**Functions:**
- Functions use `snake_case`.
- Internal helpers often use a leading underscore when they are module-local
  implementation details, such as `_load_solution`,
  `_write_no_trace_diagnostics_sidecar`, `_validate_compile_flag`, and
  `_rocm_gpu_info`.
- Validators in Pydantic models are named for the invariant they enforce, often
  with `_validate_*` or `_reject_*`, for example `_validate_entry_point`,
  `_reject_legacy_languages`, and `_reject_dangerous_flags`.
- CLI and script wrappers keep compatibility functions with explicit names, such
  as `discover_problems`, `run_cli`, and `build_cli_command`, while delegating
  to package helpers where possible.

**Variables:**
- Local variables use `snake_case`.
- Module constants use `UPPER_SNAKE_CASE`, for example
  `ENV_SNAPSHOT_ENABLE_ENV`, `PROFILE_ROCPROFV3`,
  `NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION`, and `DERIVED_EVIDENCE_SCHEMA_VERSION`.
- Private module constants use leading underscores when they are implementation
  details, for example `_DIAGNOSTIC_TAIL_LIMIT`, `_ROCM_DEVICE_NODES`, and
  `_CPP_LANGUAGES`.
- Environment variable names are stored as constants and keep the external
  spelling, such as `SOLEXECBENCH_ENV_SNAPSHOT` and
  `SOLEXECBENCH_CACHE_PATH`.

**Types:**
- Classes, dataclasses, Pydantic models, and enums use `PascalCase`, for example
  `Solution`, `BuildSpec`, `TraceRunSummary`, `ReferenceTimingResult`,
  `SupportedLanguages`, and `SupportedHardware`.
- Enum names use `PascalCase`; enum members use `UPPER_SNAKE_CASE` with string
  values that match public schema values, such as
  `SupportedLanguages.HIP_CPP = "hip_cpp"` and
  `SupportedHardware.GFX942 = "gfx942"`.
- Public schema fields use stable JSON-oriented names and avoid unnecessary
  churn because tests assert exact contracts in
  `tests/sol_execbench/test_public_contract_guardrails.py`.

## Code Style

**Formatting:**
- Python 3.12+ style with Ruff configured in `pyproject.toml`.
- Formatting command: `uv run --with ruff ruff format .`.
- Ruff `force-exclude = true` excludes `.git`, `.venv`, caches, build output,
  `data`, and `examples`.
- The codebase uses type annotations heavily, `from __future__ import
  annotations` in many modules, and explicit `Path` use for filesystem values.
- Strings are generally double-quoted in formatted Python output.

**Linting:**
- Ruff is the lint tool, configured in `pyproject.toml`.
- Lint command: `uv run --with ruff ruff check .`.
- Ruff ignores `E741`; otherwise the project largely follows Ruff defaults.
- Ty is configured for static type checking over `src` and `tests` via
  `[tool.ty.src] include = ["src", "tests"]`.
- Pre-commit is a development dependency and is used by project workflow, but
  the source conventions are primarily Ruff and Ty shaped.

## Import Organization

**Order:**
1. Future imports, when used, such as `from __future__ import annotations`.
2. Standard library imports, grouped together.
3. Third-party imports, such as `click`, `pytest`, `pydantic`, `rich`, and
   `torch`.
4. Internal `sol_execbench` imports.
5. Relative imports inside package modules.

**Grouping:**
- Blank lines separate import groups.
- Imports are usually explicit; wildcard imports are not a normal pattern.
- TYPE_CHECKING guards are used for expensive or optional type-only imports,
  for example `torch` in `src/sol_execbench/core/data/definition.py`.
- Tests sometimes adjust `sys.path` for local helper modules, as in
  `tests/sol_execbench/test_public_contract_guardrails.py`, but package imports
  are preferred.

**Path Aliases:**
- No custom Python path aliases are configured.
- Package code imports through `sol_execbench.*` or relative package imports.
- Tests share factory helpers from `tests/sol_execbench_type_helpers.py`.

## Error Handling

**Patterns:**
- Schema and contract validation raises `ValueError` through Pydantic validators,
  producing `ValidationError` at the caller boundary.
- CLI-facing failures use `click.ClickException` for actionable user errors,
  such as missing `definition.json` or `workload.jsonl` in a problem directory.
- Runtime helper failures use `RuntimeError` with specific diagnostics when
  staged files, entry points, references, or native artifacts are unavailable.
- Subprocess output is captured and bounded before persistence; CLI diagnostics
  use tail limits such as `_DIAGNOSTIC_TAIL_LIMIT`.
- Expected benchmark failures are modeled in data objects where possible rather
  than only through thrown exceptions. `TimingResult` and
  `ReferenceTimingResult` carry `failure` strings alongside latency values.

**Error Types:**
- Use `ValueError` for invalid user-provided schema values or unsafe compile
  options.
- Use `RuntimeError` for execution invariants that fail after staging, import,
  timing, or native artifact lookup.
- Use `OSError` handling around sidecar writes and filesystem persistence,
  surfacing a warning rather than crashing when diagnostic sidecar output cannot
  be written.
- Preserve exception causes for import/exec failures where useful, for example
  raising `RuntimeError(... ) from ref_err` in reference loading.

## Logging

**Framework:**
- CLI output uses Rich via `Console(stderr=True)` and Rich tables/progress.
- Scripts use normal terminal output and structured JSON sidecars where the
  output is an artifact.
- Tests generally assert returned objects, files, and subprocess output rather
  than relying on logs.

**Patterns:**
- User-facing CLI summaries are formatted through Rich tables in
  `src/sol_execbench/cli/main.py`.
- Diagnostic evidence is persisted as JSON with explicit schema versions, for
  example no-trace diagnostics and derived evidence reports.
- Low-level pure helpers avoid logging; boundary layers format or persist
  diagnostics.

## Comments

**When to Comment:**
- Comments are used for boundaries, phases, and non-obvious compatibility
  constraints, such as ROCm migration residue classification and staged
  evaluation phases.
- Short comments explain why compatibility names remain, such as PyTorch ROCm
  using `torch.cuda` namespaces.
- Avoid adding obvious comments to small helper functions.

**Docstrings:**
- Public modules, Pydantic models, dataclasses, validators, and helper functions
  commonly have concise docstrings.
- Pydantic schema fields often include docstring-style field documentation so
  generated/public schemas remain explainable.
- Tests use module and class docstrings to describe the contract under test,
  for example `TestLanguageValidation` in
  `tests/sol_execbench/core/data/test_solution.py`.

**TODO Comments:**
- No dominant TODO format was observed. Prefer issue-linked or phase-linked
  follow-up notes when adding deferred work.

## Function Design

**Size:**
- Pure helpers are kept small where possible, especially in scoring, reporting,
  runtime, and dataset support modules.
- Large CLI and dataset script modules are organized with many private helpers
  and constants rather than deeply nested logic.
- Cross-boundary behavior is usually factored into importable helpers so tests
  can exercise it without invoking the full CLI.

**Parameters:**
- Keyword-only parameters are used for options and dependency injection, for
  example `keep_staging`, `warmup`, `rep`, `time_fn`, and path-existence
  probes.
- `Path` is preferred over raw strings for filesystem APIs.
- Dataclasses are used for small immutable result objects and test case
  descriptors.

**Return Values:**
- Helpers return typed dataclasses, Pydantic models, tuples, dictionaries, or
  lists rather than loosely shaped positional data when the shape matters.
- Validation helpers return `self` in Pydantic `model_validator(mode="after")`
  methods.
- Some runtime helpers return explicit failure fields instead of throwing when a
  failure is expected diagnostic data.

## Module Design

**Exports:**
- Package entry points are exposed through `pyproject.toml` scripts:
  `sol-execbench = "sol_execbench.cli:cli"` and
  `sol-execbench-baseline = "sol_execbench.cli.baseline:cli"`.
- `src/sol_execbench/core/__init__.py` and package `__init__.py` files provide
  public import surfaces for core models and helpers.
- Scripts in `scripts/` often retain compatibility exports while delegating
  implementation to package modules under `src/sol_execbench/core/dataset/`.

**Barrel Files:**
- `__init__.py` modules are used as Python package barrels for public models and
  subpackages.
- Avoid adding circular imports through barrel files; package modules generally
  import concrete dependencies directly.

**Boundary Patterns:**
- Public JSON schemas are guarded by Pydantic models and contract tests.
- Subprocess execution is isolated through `ProblemPackager`, generated driver
  templates, and importable runtime helpers.
- ROCm-only migration boundaries are explicit: legacy CUDA/NVIDIA values are
  rejected with guidance, while unavoidable compatibility namespaces are
  documented and tested.
- Security-sensitive staging rejects absolute paths, parent traversal, response
  files, host path injection flags, runtime linker flags, and dynamic extension
  compilation during user solution execution.

---

*Convention analysis: 2026-06-04*
*Update when patterns change*
