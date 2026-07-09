---
generated_by: gsd-map-codebase
generated_on: 2026-07-09
last_mapped_commit: cc007cd3af3e5100f7d86f155a40d5e51ffb57e5
focus: quality
---

# Conventions

## Python Style

The codebase targets Python 3.12+ and is formatted/linted by Ruff. `pyproject.toml`
excludes generated or downloaded areas such as `data`, `examples`,
`__pycache__`, build artifacts, and virtualenv/cache directories. Ruff lint
currently ignores `E741`.

Use `snake_case` for modules, functions, variables, and test functions. Use
`PascalCase` for classes, enums, and Pydantic models. Enum members are uppercase
or descriptive constants depending on local style; supported schema values are
string enums in `src/sol_execbench/core/data/solution_models.py`.

## Source Headers

Many source files retain SPDX headers for Apache-2.0 and, where applicable,
upstream NVIDIA attribution. Release and provenance tooling depends on header
classification; see `scripts/internal/release/check_prerelease_readiness.py`
and provenance tests under `tests/sol_execbench/core/dataset/`.

New files should follow nearby header style and should not introduce ownership
or endorsement claims beyond project policy.

## Type And Model Style

Pydantic v2 is the schema layer. Models commonly inherit from
`BaseModelWithDocstrings` in `src/sol_execbench/core/data/base_model.py`.
Validation is implemented with `@field_validator` and `@model_validator`, as in
`src/sol_execbench/core/data/solution_models.py` and
`src/sol_execbench/core/data/solution_instance.py`.

Schema modules favor small files around one contract:

- `definition.py`, `definition_models.py`, `definition_axes.py`, and
  `definition_reference.py`.
- `solution.py`, `solution_models.py`, and `solution_instance.py`.
- `trace.py`, `workload.py`, `dtypes.py`, `shapes.py`, and `path_access.py`.

## Error Handling

CLI-facing code generally raises `click.ClickException` or exits through
central workflow handling. Evaluation subprocess failures are normalized into
runtime result objects in `src/sol_execbench/cli/evaluation/runtime.py` and
diagnostic no-trace sidecars through `src/sol_execbench/cli/evaluation/phases.py`
and related helpers.

Schema validation uses `ValueError` with actionable migration guidance, for
example rejecting `cuda_cpp` in favor of `hip_cpp`, rejecting `cuda_cflags` in
favor of `hip_cflags`, and rejecting unsafe native compile flags.

## Subprocess Boundaries

Evaluation uses a staging directory and subprocess boundary. The CLI process
does not import submitted solution code. `ProblemPackager` writes all staged
files, and `eval_driver.py` imports user code inside the staged subprocess.

Keep user-code execution, native extension imports, profiler runs, and generated
driver behavior inside this boundary. When adding features, prefer passing
bounded JSON/files through staging over importing solution modules in the CLI.

## Security Patterns

Existing security and boundary checks include:

- Source path validation rejects absolute paths and `..` in
  `SourceFile._validate_source_path()`.
- Native compile flags reject response files, host path injection, and runtime
  linker path control in `src/sol_execbench/core/data/solution_models.py`.
- Legacy CUDA/NVIDIA schema categories are rejected with ROCm migration guidance.
- `eval_driver.py` performs static source review before importing user code.
- Runtime guardrails check monkey-patching, lazy outputs, thread injection, and
  critical function integrity through `src/sol_execbench/core/bench/reward_hack/`
  and related benchmark helpers.
- No-trace diagnostics are bounded sidecars, not canonical trace JSONL.

Do not describe these checks as a hardened sandbox. Public docs explicitly say
external isolation is required for untrusted submissions.

## CLI Organization

Keep root argument parsing in `src/sol_execbench/cli/main.py` thin. New root
subcommands should normally live under `src/sol_execbench/cli/commands/`.
Evaluation-specific behavior should live under `src/sol_execbench/cli/evaluation/`.
Sidecar adapter behavior should live under `src/sol_execbench/cli/sidecars/`.

## Core Organization

Package-owned reusable logic belongs in `src/sol_execbench/core/`. Operator
scripts in `scripts/` should call core helpers where practical. The current
direction is to move durable behavior out of scripts and into importable modules
under `core/dataset/`, `core/reports/`, `core/platform/`, or `core/scoring/`.

## Documentation

Docs are treated as tested artifacts. Many tests assert specific claim
boundaries, provenance statements, schema terms, or examples. When changing
public behavior, update docs in `docs/` and related tests in
`tests/sol_execbench/core/dataset/` or `tests/sol_execbench/core/platform/`.

## Generated And Local Artifacts

Do not commit downloaded datasets, local cache directories, build outputs,
or generated benchmark traces unless they are curated doc/test fixtures.
`pyproject.toml` excludes `data` and `examples` from Ruff to avoid formatting
downloaded or intentionally embedded benchmark assets.
