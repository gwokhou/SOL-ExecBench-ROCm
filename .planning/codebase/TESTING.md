---
last_mapped_commit: dd5731c42d9f3d417acf552b3a02004cd5039df2
last_mapped_date: 2026-06-08
focus: quality
---

# Testing Patterns

## Framework And Commands

Pytest is the test framework. `pyproject.toml` configures
`addopts = "-n auto --dist loadgroup"`, so the default suite runs with
`pytest-xdist`.

Common commands:

- `uv run pytest tests/` runs the full suite.
- `uv run pytest tests/sol_execbench/test_e2e.py` runs one test file.
- `uv run pytest tests -m timing_serial -n 0` runs timing tests that are skipped by
  default.
- `uv run --with ruff ruff check .`, `uv run --with ruff ruff format .`, and
  `uv run ty check` cover lint, format, and type checks.

## Test Layout

Package tests live under `tests/sol_execbench/`. Some directories mirror source
areas directly, including `tests/sol_execbench/core/data/`,
`tests/sol_execbench/core/bench/`, and `tests/sol_execbench/driver/`.

Example workflow tests live under `tests/examples/`. `tests/examples/test_examples.py`
validates checked-in examples under `examples/`; `tests/examples/test_rocm_cli_paths.py`
covers CLI and dataset-run paths.

Docker dependency smoke tests live under `tests/docker/dependencies/` and cover HIP,
PyTorch ROCm, Triton ROCm, ROCm runtime, Python dependencies, and ROCm libraries.

Fixture data is checked in under `tests/sol_execbench/samples/`, `tests/samples/`,
and targeted fixture folders such as `tests/sol_execbench/fixtures/solar_derivation/`.
Tests commonly exercise real JSON, JSONL, and source snippets rather than broad mocks.

## Markers And Hardware Gates

Configured markers in `pyproject.toml` include `cpp`, `requires_rocm`,
`requires_rdna4`, `requires_cdna3`, and legacy `requires_cutile`.

`tests/conftest.py` registers additional marker descriptions and collection-time
skips for `timing_serial`, `requires_rocm_dev`, `requires_ck`, and
`requires_rocwmma`.

`requires_rocm` checks Linux ROCm device nodes, a ROCm PyTorch build, PyTorch-visible
GPU availability, and supported AMD architectures. `requires_rdna4` expects
`gfx12*`; `requires_cdna3` expects `gfx94*`.

Native and library examples combine markers. For example, MIOpen, CK, and rocWMMA
cases in `tests/examples/test_examples.py` use combinations of `cpp`,
`requires_rocm`, `requires_rocm_dev`, `requires_rdna4`, `requires_ck`, and
`requires_rocwmma`.

Legacy `requires_cutile` tests are skipped in this ROCm-only port and document a
compatibility boundary rather than supported NVIDIA behavior.

## Organization Patterns

Test names are behavior-focused, such as
`test_rocm_gpu_info_reports_missing_device_nodes_before_torch_probe` in
`tests/sol_execbench/test_rocm_marker_device_nodes.py` and
`test_emit_trace_jsonl_rejects_non_finite_trace_values` in
`tests/sol_execbench/core/bench/test_eval_runtime.py`.

Related schema assertions are grouped in classes, especially in files such as
`tests/sol_execbench/core/data/test_definition.py` and
`tests/sol_execbench/core/data/test_solution.py`.

Parametrization is preferred for vocabularies, invalid values, examples, hardware
targets, and matrix/report cases. E2E tests use small descriptor dataclasses such as
`Sample`, `EvilCase`, and `Example` in `tests/sol_execbench/test_e2e.py` and
`tests/examples/test_examples.py`.

Tests that cannot run concurrently use xdist grouping, commonly
`pytest.mark.xdist_group("serial")`, in subprocess, GPU, or e2e paths.

## Fixtures And Helpers

`tests/conftest.py` owns ROCm hardware and toolchain skip behavior. It probes
`/dev/kfd`, `/dev/dri`, PyTorch ROCm, detected `gfx` architecture, ROCm dev headers,
Composable Kernel headers, and rocWMMA headers during collection.

The shared `tmp_cache_dir` fixture sets `SOLEXECBENCH_CACHE_PATH` to a per-test cache
under `tmp_path`, preventing native build artifacts from leaking between tests.

`tmp_path` is the default isolation mechanism for staged problem directories,
generated manifests, sidecars, trace files, script outputs, and temporary package
trees.

Type-safe constructors live in `tests/sol_execbench_type_helpers.py` and are used
throughout the suite to build Pydantic objects while preserving local schema
expectations.

Private helpers inside test modules load samples, write JSON/JSONL fixtures, and run
subprocesses. Examples include `_load_sample` and `_run_subprocess` in
`tests/sol_execbench/test_e2e.py`.

## Mocking And Isolation

`monkeypatch` is the standard mocking tool. Tests patch environment variables,
subprocess runners, module attributes, `sys.argv`, and external loaders in files
such as `tests/sol_execbench/test_dataset_runner.py`,
`tests/sol_execbench/test_download_solexecbench.py`, and
`tests/sol_execbench/test_cli_environment_snapshot.py`.

Subprocess-heavy unit tests usually inject or monkeypatch a runner rather than
invoking real external tools. Environment and dataset tests use injected runners for
deterministic behavior.

Click commands are tested with `click.testing.CliRunner`, including contract, schema
export, toolchain routing, environment snapshot, and ROCm CLI-path coverage.

Tests that manipulate import state must restore it. `tests/sol_execbench/core/bench/test_eval_runtime.py`
uses `try`/`finally` around temporary `sys.modules` changes.

## Assertion Patterns

Validation tests assert failure type and message fragments with
`pytest.raises(..., match=...)`. This is common for schema, compatibility, download,
and derivation-contract tests.

Artifact tests read generated JSON back and assert exact fields, deterministic
ordering, status totals, source references, checksums, and claim-boundary booleans.
Representative files include `tests/sol_execbench/test_dataset_inventory_readiness.py`,
`tests/sol_execbench/test_parity_gap_report.py`,
`tests/sol_execbench/test_paper_denominator_script.py`, and
`tests/sol_execbench/test_prerelease_artifact_bundle.py`.

E2E tests assert subprocess return codes with captured stdout/stderr in the failure
message, parse canonical trace JSONL, and assert that every workload passed. See
`tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.

Documentation tests read Markdown directly to enforce public wording and claim
boundaries. Examples include `tests/sol_execbench/test_rocm_matrix_docs.py`,
`tests/sol_execbench/test_original_parity_docs.py`, and
`tests/sol_execbench/test_v1_20_evidence_quality_docs.py`.

Security and boundary tests assert that metadata parsing does not execute candidate
or reference code where execution is not intended. Relevant files include
`tests/sol_execbench/test_solar_derivation_contract.py` and
`tests/sol_execbench/test_solar_derivation_evidence.py`.

## Coverage Themes

Schema coverage is broad for definitions, workloads, solutions, traces, dtype
conversion, compatibility matrices, runtime evidence, dataset manifests, readiness,
paper-denominator reports, parity gaps, claim upgrades, trust summaries, evaluation
stability, and AMD scoring sidecars.

Execution coverage spans bench helpers, packaging, native extension staging, CLI
invocation, dataset runner behavior, e2e samples, reward-hack detection, static
kernel evidence, ROCm profiling metadata, and no-trace diagnostics.

ROCm migration coverage checks that public examples, docs, schema values, markers,
and user-facing text avoid unsupported CUDA/NVIDIA claims except where retained as
legacy compatibility boundaries. Representative files include
`tests/sol_execbench/test_rocm_library_examples.py`,
`tests/sol_execbench/test_rocm_migration_residue_audit.py`, and
`tests/sol_execbench/test_public_contract_guardrails.py`.
