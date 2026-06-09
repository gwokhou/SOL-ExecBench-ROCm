---
last_mapped_commit: dd5731c42d9f3d417acf552b3a02004cd5039df2
last_mapped_date: 2026-06-08
focus: arch
---

# Structure

## Repository Root

- `pyproject.toml`: package metadata, CLI script declarations, dependencies, UV
  indexes, pytest markers, Ruff exclusions, and type-checking source roots.
- `uv.lock`: locked dependency graph.
- `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`, and
  `THIRD_PARTY_NOTICES.txt`: public project, contribution, security, and
  compliance documents.
- `AGENTS.md` and `CLAUDE.md`: local agent and development guidance.
- `provenance.toml`: provenance policy data used by tests and release checks.
- `src/`: installable Python package.
- `tests/`: pytest suite, fixtures, samples, Docker dependency checks, and example
  workflow tests.
- `scripts/`: operational CLIs for datasets, reports, downloads, Docker helpers,
  and release checks.
- `examples/`: runnable benchmark examples grouped by implementation family.
- `docs/`: public, internal, release, schema, ROCm, and evidence documentation.
- `docker/`: ROCm container files and target matrix metadata.
- `data/`: local/downloaded benchmark assets, not package source.
- `.planning/`: GSD planning artifacts and codebase maps.

## Package Layout

`src/sol_execbench/` is the top-level package:

- `src/sol_execbench/__init__.py`: package marker.
- `src/sol_execbench/sol_score.py`: small standalone score helper.
- `src/sol_execbench/cli/`: Click command implementations.
- `src/sol_execbench/core/`: schemas, runtime helpers, dataset logic, scoring,
  diagnostics, reports, and core public exports.
- `src/sol_execbench/driver/`: staging code and generated execution/build
  templates.
- `src/sol_execbench/data/`: packaged static data, currently AMD hardware models.

## CLI

- `src/sol_execbench/cli/__init__.py`: exports the primary CLI object.
- `src/sol_execbench/cli/main.py`: main `sol-execbench` command, artifact
  loading, evaluation orchestration, sidecars, and support subcommands.
- `src/sol_execbench/cli/baseline.py`: `sol-execbench-baseline` trace comparison
  command.

## Core Data Models

`src/sol_execbench/core/data/` contains Pydantic v2 benchmark contracts:

- `base_model.py`: shared model base and constrained scalar types.
- `definition.py`: `Definition`, symbolic axes, tensor specs, and reference code
  validation.
- `workload.py`: `Workload`, input specs, safetensors/custom/random/scalar
  inputs, and tolerances.
- `solution.py`: `Solution`, `BuildSpec`, `SourceFile`, supported languages,
  bindings, hardware targets, and compile options.
- `trace.py`: `Trace`, `Evaluation`, `EvaluationStatus`, `Correctness`,
  `Performance`, and environment result payloads.
- `contract.py`: evaluator contract payload builder.
- `dtypes.py`: dtype string mapping.
- `shapes.py`: symbolic shape resolution.
- `json_utils.py`: JSON helpers.
- `__init__.py`: data-model re-export surface.

## Core Benchmark Runtime

`src/sol_execbench/core/bench/` holds runtime code imported by staged evaluation:

- `config/benchmark_config.py`: benchmark settings and clock presets.
- `config/device_config.py`: device configuration.
- `clock_lock.py`: `amd-smi` STABLE_PEAK clock lock detection and cleanup.
- `correctness.py`: output validation and numerical error statistics.
- `eval_runtime.py`: staged-driver loading, measurement, and trace helpers.
- `io.py`: workload input generation/loading and output allocation.
- `reward_hack.py`: static and runtime integrity guards.
- `rocm_profiler.py`: optional `rocprofv3` profiling wrapper.
- `static_kernel_evidence.py`: static artifact and extractor sidecars.
- `static_kernel_status.py`: static evidence status modeling.
- `stderr.py`: benign ROCm stderr filtering.
- `timing.py`: latency measurement implementation.
- `timing_policy.py`: timing policy calculations.
- `utils.py`: call/evaluation helpers.
- `__init__.py`: package marker.

## Core Dataset Modules

`src/sol_execbench/core/dataset/` contains reusable dataset and evidence logic:

- `categories.py`: dataset category validation.
- `checksums.py`: checksum helpers.
- `evidence_refs.py`: stable sidecar/reference naming.
- `execution_closure.py`: execution closure contract models and report writing.
- `inventory.py`: dataset inventory models and builders.
- `layout.py`: dataset layout inspection.
- `long_tail_exclusions.py`: long-tail exclusion sidecars and split logic.
- `low_precision.py`: architecture-specific low-precision policy and packing.
- `manifest.py`: dataset manifest generation.
- `migration.py`: source dataset migration.
- `paper_denominator.py`: paper denominator reports.
- `parity_gap.py`: parity gap reports.
- `readiness.py`: ROCm readiness classification.
- `ready_subset.py`: ready subset construction.
- `run_closure.py`: run closure records and provenance summaries.
- `run_state.py`: problem discovery, workload selection, trace mapping, and
  closure status decisions.
- `runner.py`: CLI invocation, trace parsing, summary writing, and AMD score
  report helpers.
- `sharding.py`: workload sharding and shard trace merging.
- `__init__.py`: dataset package exports.

## Core Scoring Modules

`src/sol_execbench/core/scoring/` contains AMD score and bound evidence code:

- `amd_score.py`: AMD-native score report construction.
- `amd_sol.py`: original AMD SOL bound helpers.
- `amd_sol_v2.py`: newer AMD SOL bound artifact builder.
- `amd_bound_classification.py`: bound classification models.
- `amd_bound_estimate_families.py`: bound estimate family definitions.
- `amd_bound_estimates.py`: bound estimate calculations.
- `amd_bound_graph.py`: bound graph construction.
- `amd_bound_sanity.py`: sanity checks for bound evidence.
- `amd_hardware_models.py`: hardware model loader.
- `baseline_artifact.py`: scoring baseline artifact loading.
- `solar_derivation.py`: SOLAR derivation evidence models and validation.
- `solar_derivation_status.py`: derivation status helpers.
- `__init__.py`: scoring package marker.

Packaged AMD hardware data lives under
`src/sol_execbench/data/amd_hardware_models/`, including
`src/sol_execbench/data/amd_hardware_models/gfx1200.json`.

## Other Core Modules

Focused core modules live directly under `src/sol_execbench/core/`:

- `baseline.py`: trace baseline comparison logic.
- `claim_upgrade.py`: claim upgrade report logic.
- `compatibility.py`: ROCm compatibility matrix models and classification.
- `consistency.py`: consistency report logic.
- `dependency_matrix.py`: PyTorch/ROCm dependency policy and preflight logic.
- `diagnostics.py`: diagnostic helpers.
- `docker_matrix.py`: Docker matrix target/preflight data.
- `environment.py`: ROCm/PyTorch environment probes and snapshots.
- `evaluation_stability.py`: stability report logic.
- `matrix_diff.py`: semantic diffing for matrix reports.
- `reporting.py`: shared trace/evidence summary utilities.
- `runtime_evidence.py`: runtime matrix evidence collection and CLI.
- `scoring_guardrails.py`: score interpretation guardrails.
- `toolchain.py`: ROCm toolchain registry, probes, and routing reports.
- `trust_summary.py`: trust summary report logic.
- `utils.py`: miscellaneous core helpers.
- `__init__.py`: public core re-export surface.

## Driver Directory

`src/sol_execbench/driver/` contains staging and generated templates:

- `problem_packager.py`: `ProblemPackager`, source staging, safetensors staging,
  HIP offload target detection, compile command generation, and evaluation
  command generation.
- `templates/build_ext.py`: copied into staging for native extension builds.
- `templates/eval_driver.py`: copied into staging for benchmark execution.
- `__init__.py`: driver package exports.

Template files are source-controlled Python but execute from temporary staging
directories, so path handling must be robust outside the repository root.

## Scripts

`scripts/` contains command-line tools for operations and release workflows:

- Dataset execution and migration: `scripts/run_dataset.py`,
  `scripts/download_solexecbench.py`, `scripts/download_data.sh`,
  `scripts/inspect_dataset.py`, and `scripts/run_derived_isolated.py`.
- Report generation: `scripts/report_amd_bound_sanity.py`,
  `scripts/report_claim_upgrade.py`, `scripts/report_consistency.py`,
  `scripts/report_evaluation_stability.py`,
  `scripts/report_paper_denominator.py`, `scripts/report_parity_gaps.py`, and
  `scripts/report_trust_summary.py`.
- Matrix/schema/release checks: `scripts/diff_matrix_reports.py`,
  `scripts/export_matrix_schema.py`, `scripts/check_dataset_redistribution.py`,
  `scripts/check_prerelease_readiness.py`,
  `scripts/build_prerelease_artifact_bundle.py`, and
  `scripts/release_candidate_validation.py`.
- ROCm environment helpers: `scripts/run_docker.sh` and
  `scripts/setup_rocm_clock_sudoers.py`.

## Examples

`examples/` contains runnable problem directories. Each example generally
contains `definition.json`, `workload.jsonl`, a `solution_*.json`, and source
files such as `kernel.py`, `reference.py`, `kernel.hip`, `kernel.h`, or
`main.cpp`.

Current families include:

- `examples/pytorch/`
- `examples/triton/`
- `examples/hip_cpp/`
- `examples/hipblas/`
- `examples/miopen/`
- `examples/ck/`
- `examples/rocwmma/`
- Legacy/parity examples under `examples/cudnn/`, `examples/cutlass/`,
  `examples/cutile/`, and `examples/cute_dsl/`.

`examples/` is excluded from Ruff checks in `pyproject.toml`.

## Tests

`tests/` is organized by concern:

- `tests/sol_execbench/core/data/`: schema model tests.
- `tests/sol_execbench/core/bench/`: benchmark runtime helper tests.
- `tests/sol_execbench/driver/`: staging and template tests.
- `tests/sol_execbench/test_*.py`: broader CLI, dataset, scoring, report,
  policy, documentation, and guardrail tests.
- `tests/sol_execbench/samples/`: sample benchmark problems used by tests.
- `tests/sol_execbench/fixtures/`: structured fixtures, especially SOLAR
  derivation cases.
- `tests/examples/`: example workflow and ROCm CLI path tests.
- `tests/docker/dependencies/`: Docker/ROCm dependency smoke tests.
- `tests/conftest.py`: pytest configuration and marker behavior.
- `tests/sol_execbench_type_helpers.py`: type helper tests.

Pytest markers in `pyproject.toml` include `cpp`, `requires_rocm`,
`requires_rdna4`, `requires_cdna3`, and legacy `requires_cutile`.

## Documentation

`docs/` contains:

- User/dev docs: `docs/GETTING-STARTED.md`, `docs/CONFIGURATION.md`,
  `docs/COOKBOOK.md`, `docs/DEVELOPMENT.md`, `docs/RESEARCHER-GUIDE.md`,
  `docs/ARCHITECTURE.md`, and `docs/TESTING.md`.
- Schema docs: `docs/definition.md`, `docs/workload.md`, `docs/solution.md`,
  and `docs/trace.md`.
- ROCm docs: `docs/rocm.md`, `docs/rocm_libraries.md`,
  `docs/rocm_timing.md`, and `docs/rocm_toolchain_routing.md`.
- Evidence/release docs: `docs/CLAIMS.md`, `docs/research_preview.md`,
  `docs/original_parity.md`, `docs/provenance.md`,
  `docs/prerelease_readiness.md`, `docs/prerelease_artifact_bundle.md`,
  `docs/release_candidate_validation.md`, and versioned release notes/checklists.
- Demo evidence: `docs/examples/v1_19_evidence/` and
  `docs/examples/v1_20_evidence_quality/`.
- Internal validation notes: `docs/internal/`.
- Release drafts: `docs/releases/`.

## Docker And Data

- `docker/Dockerfile`: ROCm container definition.
- `docker/entrypoint.sh`: container startup/runtime setup.
- `docker/rocm-targets.json`: supported Docker matrix target metadata.
- `data/`: local benchmark assets such as downloaded SOL ExecBench data and
  FlashInfer trace blobs.

Treat `data/` as local artifact storage, not package architecture.

## Generated And Local Artifacts

Generated/cache paths should not drive architecture decisions:

- `.pytest_cache/`
- `.ruff_cache/`
- `__pycache__/`
- `*.pyc`
- `build/`
- `dist/`
- downloaded or generated trees under `data/`
- temporary staging directories produced by CLI or dataset runs
