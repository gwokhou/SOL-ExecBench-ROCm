---
last_mapped_commit: f4de6692ee7468e150c112e0cbdcc8842dd0c709
mapped_at: 2026-06-06
---

# Coding Conventions

## Style Baseline

The repository is a Python 3.12+ package rooted at `src/sol_execbench/`, with scripts in `scripts/` and tests in `tests/`. Formatting and linting are governed by Ruff in `pyproject.toml`: generated/downloaded content, `data/`, and `examples/` are excluded; `E741` is ignored. Type checking is configured through `tool.ty.src` for `src` and `tests`.

Most source files use SPDX headers. Files retained from the original implementation often include both NVIDIA and ROCm-port contributor copyright lines, for example `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/data/solution.py`, and `tests/examples/test_examples.py`. Newer ROCm-only files commonly use only the contributor SPDX lines, for example `scripts/check_prerelease_readiness.py` and `src/sol_execbench/core/dataset/readiness.py`.

`from __future__ import annotations` is common in newer modules and tests, especially CLI, dataset, environment, scoring, and script code such as `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/dataset/runner.py`, and `scripts/release_candidate_validation.py`.

## Naming

Use `snake_case` for modules, functions, variables, helper functions, and fixture names. Internal helpers are prefixed with `_`, such as `_load_solution` in `src/sol_execbench/cli/main.py`, `_validate_compile_flag` in `src/sol_execbench/core/data/solution.py`, and `_missing_rocm_device_nodes` in `tests/conftest.py`.

Use `PascalCase` for classes, dataclasses, Pydantic models, and test classes. Examples include `BuildSpec`, `CompileOptions`, and `SupportedHardware` in `src/sol_execbench/core/data/solution.py`, `EnvironmentSnapshot` in `src/sol_execbench/core/environment.py`, and test groupings like `TestLanguageValidation` in `tests/sol_execbench/core/data/test_solution.py`.

Enum classes use `PascalCase`; enum members use uppercase names with string values. ROCm vocabulary is explicit, as in `SupportedLanguages.HIP_CPP`, `SupportedLanguages.MIOPEN`, `SupportedHardware.GFX1200`, and `SupportedHardware.GFX942` in `src/sol_execbench/core/data/solution.py`.

Constants are uppercase at module scope, including CLI/env constants such as `ENV_SNAPSHOT_ENABLE_ENV`, `PROFILE_ROCPROFV3`, and `NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION` in `src/sol_execbench/cli/main.py`, and dataset/test constants like `_SAMPLES_DIR` in `tests/sol_execbench/test_e2e.py`.

## Models And Data Contracts

Public schema objects use Pydantic v2. Shared aliases and base behavior live in `src/sol_execbench/core/data/base_model.py`, where `BaseModelWithDocstrings` enables `use_attribute_docstrings=True`, and aliases like `NonEmptyString` and `NonNegativeInt` wrap `Annotated` plus `Field` constraints.

Schema validation is concentrated in Pydantic validators. `src/sol_execbench/core/data/solution.py` uses `@field_validator` and `@model_validator` to reject legacy CUDA/NVIDIA language values, enforce entry point suffixes, validate source paths, and block unsafe native compile flags. Tests assert validation messages, so error text is part of the practical contract.

Report and evidence objects generally expose deterministic JSON methods using `model_dump(mode="json")`, `json.dumps(..., sort_keys=True)`, and often `indent=2` plus a trailing newline. This pattern appears in `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/readiness.py`, `src/sol_execbench/core/dataset/ready_subset.py`, `src/sol_execbench/core/dataset/paper_denominator.py`, and `src/sol_execbench/core/claim_upgrade.py`.

Dataclasses are used for internal descriptors and simple immutable calculation inputs where Pydantic validation is not needed. Examples include frozen dataclasses in `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_hardware_models.py`, `src/sol_execbench/core/dataset/low_precision.py`, and test descriptors in `tests/examples/test_examples.py`.

## File And JSON Handling

Use `pathlib.Path` consistently for filesystem paths. CLI options use Click `path_type=Path` in `src/sol_execbench/cli/main.py`, scripts resolve repository roots with `Path(__file__).resolve()`, and tests use `tmp_path` for isolated staging.

JSON and JSONL parsing is straightforward and explicit: `json.loads(path.read_text())` for JSON files, and line-by-line parsing with blank-line skips for JSONL. Examples include `_load_workloads` in `src/sol_execbench/cli/main.py`, `_load_sample` in `tests/sol_execbench/test_e2e.py`, and dataset runner logic in `src/sol_execbench/core/dataset/runner.py`.

When writing machine-readable artifacts, prefer sorted deterministic output and explicit parent creation. Sidecar writers in `src/sol_execbench/cli/main.py` create parent directories before writing, while dataset/report modules such as `src/sol_execbench/core/dataset/manifest.py` and `src/sol_execbench/core/dataset/parity_gap.py` serialize stable JSON for reproducible tests and release artifacts.

## CLI And Subprocess Patterns

The primary CLI is Click-based in `src/sol_execbench/cli/main.py`, with Rich used for user-facing tables and status messages. CLI failure paths raise `click.ClickException` for user errors, for example missing problem inputs or unsupported output modes.

Subprocess calls use explicit argument lists, `capture_output=True`, `text=True`, and timeouts. The CLI's `_run_evaluation_command` and compile/evaluate phases in `src/sol_execbench/cli/main.py` follow this pattern, as do tests and scripts such as `tests/examples/test_examples.py`, `tests/sol_execbench/test_dependency_matrix_cli.py`, and `scripts/run_dataset.py`.

Subprocess output that may be diagnostic or noisy is bounded or filtered. `src/sol_execbench/cli/main.py` stores no-trace diagnostics with tail limits, filters benign ROCm stderr through `filter_benign_rocm_stderr`, and records sidecars rather than treating all auxiliary failures as benchmark failures.

Scripts are usually importable modules with a `main(...)` function and a final `raise SystemExit(main())`, as in `scripts/export_matrix_schema.py`, `scripts/report_trust_summary.py`, `scripts/check_dataset_redistribution.py`, and `scripts/release_candidate_validation.py`.

## Error Handling

Schema and contract violations generally raise `ValueError` or Pydantic `ValidationError`. Examples include dtype validation in `src/sol_execbench/core/data/dtypes.py`, solution schema validation in `src/sol_execbench/core/data/solution.py`, hardware model loading in `src/sol_execbench/core/scoring/amd_hardware_models.py`, and matrix contracts in `src/sol_execbench/core/runtime_evidence.py`.

Runtime/evaluation boundary failures generally raise `RuntimeError` with contextual messages. Examples include missing staged files and failed dynamic imports in `src/sol_execbench/core/bench/eval_runtime.py`, missing outputs in `src/sol_execbench/core/bench/io.py`, and native build template errors in `src/sol_execbench/driver/templates/build_ext.py`.

CLI-visible misuse raises `click.ClickException` in `src/sol_execbench/cli/main.py`; script argument misuse commonly raises `SystemExit` with a concise message, as in `scripts/run_dataset.py` and `scripts/release_candidate_validation.py`.

Non-critical evidence collection is deliberately nonfatal. Environment snapshots, profiling metadata, static evidence sidecars, optional probes, and missing external tools often catch `OSError`, `FileNotFoundError`, `ImportError`, `RuntimeError`, or broad `Exception`, then return `None`, record a failed/unsupported sidecar, or print a yellow diagnostic. See `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/runtime_evidence.py`, and `src/sol_execbench/core/bench/rocm_profiler.py`.

Security-sensitive validation is fail-closed. `src/sol_execbench/core/data/solution.py` rejects absolute paths, parent traversal, response files, sysroot/path-injection flags, rpaths, and runtime linker overrides. `scripts/download_solexecbench.py` rejects unsafe remote problem names, and dataset readiness checks in `src/sol_execbench/core/dataset/readiness.py` block safetensors paths outside the dataset root.

## ROCm Port Conventions

ROCm-only vocabulary is preferred in new code and tests: HIP/C++, hipBLAS, MIOpen, Composable Kernel, rocWMMA, RDNA 4, CDNA 3, `gfx1200`, and `gfx94*`. Legacy CUDA/NVIDIA values are either rejected with migration guidance or retained only in compatibility tests and historical examples.

Validation claims are carefully bounded. Matrix, evidence, and release code separates target hardware, observed evidence, diagnostic evidence, container validation, native host validation, and claim authority. Relevant modules include `src/sol_execbench/core/runtime_evidence.py`, `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/docker_matrix.py`, and tests like `tests/sol_execbench/test_matrix_claim_guardrails.py`.

Hardware-gated behavior should use existing marker and probe conventions rather than assuming a GPU. `tests/conftest.py` probes `/dev/kfd`, `/dev/dri`, PyTorch ROCm availability, ROCm dev headers, CK headers, rocWMMA headers, and detected `gfx` architecture before adding skip markers.
