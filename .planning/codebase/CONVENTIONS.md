---
last_mapped_commit: dd5731c42d9f3d417acf552b3a02004cd5039df2
last_mapped_date: 2026-06-08
focus: quality
---

# Coding Conventions

## Style Baseline

The project is a Python 3.12+ package rooted at `src/sol_execbench/`, with CLIs in
`src/sol_execbench/cli/`, scripts in `scripts/`, examples in `examples/`, and tests
in `tests/`.

Ruff is the formatter and linter configured in `pyproject.toml`. It force-excludes
generated/cache/build data plus `data/` and `examples/`, and currently ignores
`E741`. Type checking is configured with `ty` over `src` and `tests`.

Pre-commit hooks in `.pre-commit-config.yaml` run `uv run --locked ruff check --fix`,
`uv run --locked ruff format`, a DCO sign-off check, and `uv run --locked ty check`
on pre-push.

Most source and test files use SPDX headers. Original-derived files often include
both NVIDIA and ROCm-port contributor copyright lines, such as
`src/sol_execbench/cli/main.py` and `tests/examples/test_examples.py`; newer
ROCm-only files commonly use the contributor SPDX lines only, such as
`src/sol_execbench/core/bench/eval_runtime.py`.

## Python Style

Use `from __future__ import annotations` in new Python modules unless there is a
local reason not to. This is common in CLI, bench, dataset, script, and test code.

Use `pathlib.Path` for filesystem paths. Load JSON with explicit
`json.loads(path.read_text())`, parse JSONL line by line while skipping blank lines,
and write deterministic artifacts with explicit parent creation where needed.

Keep helpers small and local when they serve one module. Internal helpers are
prefixed with `_`, for example `_load_solution` in `src/sol_execbench/cli/main.py`,
`_safe_module_part` in `src/sol_execbench/core/bench/eval_runtime.py`, and
`_missing_rocm_device_nodes` in `tests/conftest.py`.

Prefer dependency injection for testable external behavior. Existing code accepts
runner/time/probe callables in places such as `src/sol_execbench/core/bench/eval_runtime.py`
and environment/report modules rather than hard-wiring all subprocess or timing calls.

## Naming

Use `snake_case` for modules, functions, variables, fixtures, and private helpers.
Use `PascalCase` for classes, dataclasses, Pydantic models, and test grouping
classes.

Enum classes use `PascalCase`; enum members use uppercase names with string values.
ROCm vocabulary is explicit in public enums and constants, for example
`SupportedLanguages.HIP_CPP`, `SupportedLanguages.MIOPEN`, `SupportedHardware.GFX1200`,
and `SupportedHardware.GFX942` in `src/sol_execbench/core/data/solution.py`.

Module constants are uppercase. CLI and evidence constants such as
`ENV_SNAPSHOT_ENABLE_ENV`, `PROFILE_ROCPROFV3`, and
`NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION` live in `src/sol_execbench/cli/main.py`;
test sample roots such as `_SAMPLES_DIR` live near their users.

## Models And Contracts

Public schemas use Pydantic v2. Shared base behavior and constrained aliases live in
`src/sol_execbench/core/data/base_model.py`, including `BaseModelWithDocstrings`,
`NonEmptyString`, and `NonNegativeInt`.

Schema validation is concentrated in `@field_validator` and `@model_validator`
methods. `src/sol_execbench/core/data/definition.py` validates reference code,
axis usage, and custom input entrypoints; `src/sol_execbench/core/data/solution.py`
validates language, hardware, entry point, source path, and native compile options.

Validation messages are treated as practical contract surface. Tests commonly assert
`pytest.raises(..., match=...)`, so new errors should be specific and stable enough
for users and tests.

Use dataclasses for internal descriptors and calculation inputs when Pydantic
validation is unnecessary. Examples include runtime timing results in
`src/sol_execbench/core/bench/eval_runtime.py` and e2e descriptor objects in
`tests/sol_execbench/test_e2e.py`.

## CLI And Scripts

The main user CLI is Click-based in `src/sol_execbench/cli/main.py`, with Rich for
tables and progress display. User misuse generally raises `click.ClickException`;
script misuse commonly raises `SystemExit` with a concise message.

Subprocess calls use explicit argument lists, `capture_output=True`, `text=True`,
and timeouts. This pattern appears in CLI evaluation paths, scripts such as
`scripts/run_dataset.py`, and e2e tests.

Scripts are usually importable modules with a `main(...)` function and a final
`raise SystemExit(main())`, as in `scripts/export_matrix_schema.py`,
`scripts/report_trust_summary.py`, and `scripts/release_candidate_validation.py`.

## Error Handling

Use `ValueError` or Pydantic validation errors for schema and contract violations.
Use `RuntimeError` for runtime boundary failures such as missing staged files,
failed dynamic imports, or native build errors.

CLI-visible failures should be converted to `click.ClickException` where possible.
Diagnostic-only failures should be bounded and recorded rather than allowed to
overwrite benchmark truth.

Non-critical evidence collection is intentionally nonfatal. Environment snapshots,
rocprofv3 metadata, static evidence, and optional external probes often return
`None`, emit a warning, or write an unsupported/failed sidecar instead of failing
the run.

Security-sensitive validation is fail-closed. `src/sol_execbench/core/data/solution.py`
rejects absolute paths, parent traversal, response files, unsafe compile/link flags,
rpaths, and runtime linker overrides. Dataset and download code similarly checks
unsafe names and paths.

## ROCm Port Rules

Prefer ROCm-only vocabulary in new code: HIP/C++, hipBLAS, MIOpen, Composable
Kernel, rocWMMA, RDNA 4, CDNA 3, `gfx1200`, and `gfx94*`.

Legacy CUDA/NVIDIA values are retained only for compatibility boundaries, migration
tests, historical examples, or explicit rejection paths. Do not add new CUDA support
surfaces unless the project scope changes.

Hardware behavior must be marker/probe-gated instead of assumed. `tests/conftest.py`
is the central place for ROCm device-node, PyTorch ROCm, architecture, ROCm dev
header, CK, and rocWMMA detection.

Validation claims should separate target support, observed evidence, diagnostic
evidence, container validation, host validation, and claim authority. Relevant
modules include `src/sol_execbench/core/runtime_evidence.py`,
`src/sol_execbench/core/dependency_matrix.py`, and
`src/sol_execbench/core/docker_matrix.py`.
