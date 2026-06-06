---
last_mapped_commit: f4de6692ee7468e150c112e0cbdcc8842dd0c709
mapped_at: 2026-06-06
---

# Testing Patterns

## Framework And Configuration

Pytest is the test framework. `pyproject.toml` sets `addopts = "-n auto --dist loadgroup"`, so the suite runs with `pytest-xdist` by default. Tests that cannot safely run concurrently use `pytest.mark.xdist_group("serial")`, especially GPU/e2e paths in `tests/sol_execbench/test_e2e.py`, `tests/examples/test_examples.py`, and `tests/examples/test_rocm_cli_paths.py`.

Configured markers in `pyproject.toml` include `cpp`, `requires_rocm`, `requires_rdna4`, `requires_cdna3`, and legacy `requires_cutile`. `tests/conftest.py` adds additional marker descriptions and collection-time skip logic for `timing_serial`, `requires_rocm_dev`, `requires_ck`, and `requires_rocwmma`.

Default command patterns are:

- `uv run pytest tests/` for the full suite.
- `uv run pytest tests/sol_execbench/test_e2e.py` for one file.
- `uv run pytest tests -m timing_serial -n 0` for timing tests that are skipped by default.
- ROCm/GPU examples are marker-gated and require visible ROCm hardware and device nodes.

## Test Layout

Package tests live under `tests/sol_execbench/`, mirroring source areas where useful. Data schema tests are in `tests/sol_execbench/core/data/`, bench helper tests are in `tests/sol_execbench/core/bench/`, and driver tests are in `tests/sol_execbench/driver/`.

Example workflow tests live under `tests/examples/`. `tests/examples/test_examples.py` validates checked-in examples under `examples/`, while `tests/examples/test_rocm_cli_paths.py` covers ROCm CLI and dataset-run flows.

Docker dependency smoke tests live in `tests/docker/dependencies/`, including ROCm runtime, HIP compiler, PyTorch ROCm, Triton ROCm, and ROCm library checks.

Fixture data is checked in under `tests/sol_execbench/samples/`, `tests/samples/`, and focused fixture directories such as `tests/sol_execbench/fixtures/solar_derivation/`. Tests commonly assert against real JSON, JSONL, and source snippets rather than broad synthetic mocks.

## Test Naming And Organization

Test names are descriptive and behavior-focused, for example `test_rejects_invalid_solution_schema`, `test_run_cli_writes_log_for_timeout`, `test_matrix_report_rejects_negative_status_counts`, and `test_builder_does_not_accept_or_execute_candidate_solution_code`.

Related assertions are often grouped in classes for schema modules, such as `TestLanguageValidation`, `TestEntryPointSuffixValidation`, and `TestHardwareAndCompileOptions` in `tests/sol_execbench/core/data/test_solution.py`, or driver groups in `tests/sol_execbench/driver/test_problem_packager.py`.

Parametrization is preferred for enum vocabularies, invalid values, examples, hardware targets, and compatibility matrices. Descriptor dataclasses such as `Sample`, `EvilCase`, and `Example` in `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py` make e2e parameter sets readable.

## Fixtures And Helpers

`tests/conftest.py` owns hardware and toolchain skip behavior. It probes Linux device nodes, PyTorch ROCm, detected `gfx` architecture, ROCm dev headers, CK headers, and rocWMMA headers, then marks tests during collection instead of failing at runtime.

The shared `tmp_cache_dir` fixture in `tests/conftest.py` sets `SOLEXECBENCH_CACHE_PATH` to a per-test cache under `tmp_path`, preventing native build artifacts from leaking between tests.

Tests use `tmp_path` heavily for staged problem directories, generated manifests, sidecars, trace files, and script outputs. Examples include `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/test_dataset_runner.py`, `tests/sol_execbench/test_cli_environment_snapshot.py`, and `tests/sol_execbench/test_dataset_inventory_readiness.py`.

Type-safe helper constructors live in `tests/sol_execbench_type_helpers.py` and are used throughout tests to construct Pydantic objects while preserving local schema expectations.

Private helper functions inside test modules prepare sample problems, write JSON/JSONL fixtures, and normalize subprocess execution. Examples include `_load_sample` and `_run_subprocess` in `tests/sol_execbench/test_e2e.py`, `_load_example` in `tests/examples/test_examples.py`, and dataset-writing helpers in `tests/examples/test_rocm_cli_paths.py`.

## Mocking And Isolation

`monkeypatch` is the standard mocking tool. Tests patch subprocess runners, environment variables, module attributes, `sys.argv`, and external loaders. Representative files include `tests/sol_execbench/test_dataset_runner.py`, `tests/sol_execbench/test_download_solexecbench.py`, `tests/sol_execbench/test_cli_environment_snapshot.py`, `tests/examples/test_rocm_cli_paths.py`, and script tests for report generation.

Subprocess-heavy unit tests usually inject or monkeypatch a runner rather than invoking real external tools. For example, `tests/sol_execbench/test_dataset_runner.py` patches `runner.subprocess.run`; `tests/sol_execbench/test_environment_snapshot.py` uses injected runners; release validation tests in `tests/sol_execbench/test_release_candidate_validation.py` monkeypatch command execution and artifact handling.

Click commands are tested with `click.testing.CliRunner`, for example in `tests/sol_execbench/test_contract.py`, `tests/sol_execbench/test_matrix_schema_export.py`, `tests/sol_execbench/test_toolchain_routing.py`, `tests/sol_execbench/test_cli_environment_snapshot.py`, and `tests/examples/test_rocm_cli_paths.py`.

Output capture uses pytest fixtures such as `capsys` for script output assertions, as in `tests/sol_execbench/test_dataset_runner.py` and `tests/sol_execbench/test_dataset_redistribution_policy.py`.

Tests that intentionally manipulate import state clean up after themselves. `tests/sol_execbench/core/bench/test_eval_runtime.py` temporarily overrides `sys.modules` to verify staged module isolation, then restores the previous module state in `finally` blocks.

## Assertion Patterns

Validation tests assert both failure type and message fragments using `pytest.raises(..., match=...)`. This is common in schema and contract tests such as `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/test_rocm_compatibility_matrix.py`, `tests/sol_execbench/test_solar_derivation_evidence.py`, and `tests/sol_execbench/test_download_solexecbench.py`.

Artifact tests read generated JSON back and assert exact fields, deterministic ordering, status totals, source refs, checksums, and claim-boundary booleans. See `tests/sol_execbench/test_dataset_inventory_readiness.py`, `tests/sol_execbench/test_parity_gap_report.py`, `tests/sol_execbench/test_paper_denominator_script.py`, `tests/sol_execbench/test_amd_bound_sanity_script.py`, and `tests/sol_execbench/test_prerelease_artifact_bundle.py`.

E2E tests assert subprocess return codes with captured stdout/stderr in failure messages, then parse canonical trace JSONL and assert every workload passed. This pattern is visible in `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.

Documentation tests enforce public wording and claim boundaries by reading Markdown files directly. Examples include `tests/sol_execbench/test_rocm_matrix_docs.py`, `tests/sol_execbench/test_original_parity_docs.py`, `tests/sol_execbench/test_research_release_docs.py`, `tests/sol_execbench/test_public_prerelease_docs.py`, and `tests/sol_execbench/test_v1_20_evidence_quality_docs.py`.

Security and boundary tests assert that code is not executed when parsing metadata. Examples include `tests/sol_execbench/test_solar_derivation_contract.py`, which verifies fixture loading does not execute reference text, and `tests/sol_execbench/test_solar_derivation_evidence.py`, which checks derivation builders do not accept or execute candidate solution code.

## Hardware And Slow Test Patterns

Hardware requirements are marker-gated instead of assumed. `requires_rocm` needs a ROCm GPU visible through PyTorch and supported architecture; `requires_rdna4` expects `gfx12*`; `requires_cdna3` expects `gfx94*`; `requires_rocm_dev` expects HIP headers; `requires_ck` and `requires_rocwmma` expect corresponding ROCm library headers.

Legacy `requires_cutile` tests are skipped in this ROCm-only port. Tests retain this marker only to document compatibility boundaries and avoid accidentally treating NVIDIA-only paths as ROCm-supported.

Native HIP/C++ and library examples use `cpp` plus more specific markers. `tests/examples/test_examples.py` marks MIOpen, CK, and rocWMMA cases with combinations of `cpp`, `requires_rocm`, `requires_rocm_dev`, `requires_rdna4`, `requires_ck`, or `requires_rocwmma`.

GPU timing tests use `timing_serial` and are skipped unless explicitly selected with `-m timing_serial`. This prevents routine CPU-safe suites from depending on noisy wall-clock or GPU timing behavior.

## Coverage Themes

Schema coverage is broad for public contracts: data definitions, workloads, solutions, traces, dtype conversion, ROCm compatibility matrices, runtime evidence, dataset manifests, readiness, ready subsets, paper denominator reports, parity gap reports, claim upgrades, trust summaries, evaluation stability, and AMD scoring sidecars.

Execution coverage spans unit-level bench helpers, packaging, native extension staging, CLI invocation, dataset runner behavior, e2e samples, reward-hack detection, static kernel evidence, ROCm profiling metadata, and no-trace diagnostics.

ROCm migration coverage explicitly checks that public examples, docs, schema values, markers, and user-facing text avoid unsupported CUDA/NVIDIA claims except where retained as legacy compatibility boundaries. Relevant files include `tests/sol_execbench/test_rocm_library_examples.py`, `tests/sol_execbench/test_rocm_migration_residue_audit.py`, `tests/sol_execbench/test_rocm_test_suite_audit.py`, and `tests/sol_execbench/test_public_contract_guardrails.py`.

Release and evidence quality coverage is artifact-oriented. Tests exercise scripts such as `scripts/report_paper_denominator.py`, `scripts/report_parity_gaps.py`, `scripts/report_amd_bound_sanity.py`, `scripts/report_evaluation_stability.py`, `scripts/report_claim_upgrade.py`, `scripts/check_prerelease_readiness.py`, `scripts/release_candidate_validation.py`, and `scripts/build_prerelease_artifact_bundle.py`.
