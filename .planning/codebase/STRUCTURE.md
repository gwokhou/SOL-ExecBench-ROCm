---
last_mapped_commit: f4de6692ee7468e150c112e0cbdcc8842dd0c709
mapped_at: 2026-06-06
---

# Structure

## Repository Root

- `pyproject.toml`: package metadata, Python version constraint, runtime
  dependencies, UV indexes, CLI scripts, pytest markers, Ruff exclusions, and
  type-checking source roots.
- `uv.lock`: locked dependency graph.
- `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`, and
  `THIRD_PARTY_NOTICES.txt`: public project and compliance documents.
- `AGENTS.md` and `CLAUDE.md`: local agent/development guidance.
- `provenance.toml`: provenance policy data used by tests and release checks.
- `src/`: installable Python package.
- `tests/`: pytest suite, sample problems, Docker dependency tests, and example
  workflow tests.
- `scripts/`: operational CLIs and release/report helpers.
- `examples/`: runnable benchmark examples grouped by implementation family.
- `docs/`: user, researcher, architecture, release, and evidence documentation.
- `docker/`: ROCm container build/runtime support.
- `data/`: downloaded or local benchmark assets; not package source.
- `.planning/`: GSD planning artifacts and codebase maps.

## Package Layout

`src/sol_execbench/` is the top-level package.

- `src/sol_execbench/__init__.py`: package marker.
- `src/sol_execbench/sol_score.py`: small standalone scoring helper.
- `src/sol_execbench/cli/`: Click command implementations.
- `src/sol_execbench/core/`: data schemas, runtime helpers, dataset logic,
  scoring, diagnostics, reports, and public core exports.
- `src/sol_execbench/driver/`: staging and generated driver templates.
- `src/sol_execbench/data/`: packaged static data, currently AMD hardware model
  JSON.

## CLI Files

- `src/sol_execbench/cli/__init__.py`: exposes the CLI group/command object.
- `src/sol_execbench/cli/main.py`: main `sol-execbench` CLI. Contains the
  evaluation command, `contract`, `doctor`, `toolchain`, and `dataset migrate`
  subcommands.
- `src/sol_execbench/cli/baseline.py`: `sol-execbench-baseline` CLI for comparing
  candidate traces with baseline trace JSONL files.

Naming convention in this layer: command functions are private helpers such as
`_evaluate_cli`, `_contract_cli`, `_doctor_cli`, and `_toolchain_cli`, then wired
into the exported Click command object. Loader and sidecar helpers use
descriptive `_load_*`, `_write_*`, and `_*_path` names.

## Core Data Models

`src/sol_execbench/core/data/` contains Pydantic v2 models and schema utilities.

- `base_model.py`: shared base model and constrained scalar types.
- `definition.py`: `Definition`, symbolic axes, tensor specs, and reference code
  validation.
- `workload.py`: `Workload`, input specs, random/scalar/custom/safetensors inputs,
  and tolerances.
- `solution.py`: `Solution`, `BuildSpec`, `SourceFile`, supported languages,
  AMD hardware targets, bindings, and compile options.
- `trace.py`: `Trace`, `Evaluation`, `EvaluationStatus`, `Correctness`,
  `Performance`, and result environment.
- `contract.py`: evaluator contract payload builder.
- `dtypes.py`: dtype string to Torch dtype mapping.
- `shapes.py`: symbolic shape expression resolution.
- `json_utils.py`: JSON helper functions.
- `__init__.py`: data-model re-export surface.

Model classes use `PascalCase`; modules and helper functions use `snake_case`.
Enum classes use `PascalCase` and members use uppercase names with string
values.

## Core Benchmark Runtime

`src/sol_execbench/core/bench/` holds code used by host-side orchestration and
the staged evaluation driver.

- `config/benchmark_config.py`: benchmark settings.
- `config/device_config.py`: device-related configuration.
- `clock_lock.py`: GPU clock lock detection.
- `correctness.py`: output shape/dtype checks and numerical error statistics.
- `eval_runtime.py`: staged-driver helpers for loading code, running functions,
  measuring latency, and emitting traces.
- `io.py`: input generation/loading, output allocation, and safetensors path
  environment handling.
- `reward_hack.py`: static and runtime benchmark integrity checks.
- `rocm_profiler.py`: optional `rocprofv3` profiling integration.
- `static_kernel_evidence.py`: static source/kernel evidence collection.
- `static_kernel_status.py`: status modeling for static evidence.
- `stderr.py`: benign ROCm stderr filtering.
- `timing.py`: latency measurement implementation.
- `timing_policy.py`: timing policy calculations.
- `utils.py`: utility helpers for calling solutions and building evaluations.
- `__init__.py`: package marker.

Files in this directory should stay importable from staged execution contexts,
because `src/sol_execbench/driver/templates/eval_driver.py` imports them at
runtime.

## Core Dataset Modules

`src/sol_execbench/core/dataset/` contains reusable dataset execution and
evidence logic.

- `categories.py`: known dataset category policy.
- `checksums.py`: checksum helpers.
- `evidence_refs.py`: stable sidecar and relative reference helpers.
- `execution_closure.py`: execution closure contract helpers.
- `inventory.py`: dataset inventory helpers.
- `layout.py`: dataset layout validation.
- `low_precision.py`: architecture-specific low precision skip policy.
- `manifest.py`: dataset manifest generation and metadata.
- `migration.py`: source dataset migration helpers.
- `paper_denominator.py`: paper denominator reporting.
- `parity_gap.py`: parity gap report helpers.
- `readiness.py`: readiness metadata handling.
- `ready_subset.py`: ready-subset metadata handling.
- `run_closure.py`: run closure records, totals, provenance, and report writing.
- `run_state.py`: problem discovery, workload selection, trace mapping, and
  closure status decisions.
- `runner.py`: importable CLI invocation, trace parsing, summary writing, and
  AMD score report helpers.
- `sharding.py`: workload sharding helpers.
- `__init__.py`: dataset package exports.

Operational script compatibility exists in `scripts/run_dataset.py`, but new
reusable behavior should usually be placed here first.

## Core Scoring Modules

`src/sol_execbench/core/scoring/` contains AMD-specific score and evidence code.

- `amd_score.py`: AMD-native score report construction.
- `amd_sol.py`: original AMD SOL bound helpers and default hardware models.
- `amd_sol_v2.py`: newer AMD SOL bound artifact builder.
- `amd_bound_classification.py`: bound classification models.
- `amd_bound_estimate_families.py`: bound estimate family definitions.
- `amd_bound_estimates.py`: bound estimate calculations.
- `amd_bound_graph.py`: bound graph construction.
- `amd_bound_sanity.py`: sanity checks for bound evidence.
- `amd_hardware_models.py`: hardware model loader.
- `baseline_artifact.py`: scoring baseline artifact loader.
- `solar_derivation.py`: SOLAR derivation evidence models and validation.
- `solar_derivation_status.py`: derivation status helpers.
- `__init__.py`: scoring package marker.

Packaged AMD hardware data lives in:

- `src/sol_execbench/data/amd_hardware_models/gfx1200.json`
- `src/sol_execbench/data/amd_hardware_models/__init__.py`

## Other Core Modules

Several focused modules live directly under `src/sol_execbench/core/`:

- `baseline.py`: baseline trace comparison logic.
- `claim_upgrade.py`: claim upgrade report logic.
- `compatibility.py`: ROCm compatibility policy helpers.
- `consistency.py`: consistency report logic.
- `dependency_matrix.py`: dependency matrix construction and CLI-facing policy.
- `diagnostics.py`: diagnostic helpers.
- `docker_matrix.py`: Docker matrix target/preflight data.
- `environment.py`: ROCm/PyTorch environment probes and snapshots.
- `evaluation_stability.py`: stability report logic.
- `matrix_diff.py`: semantic diffing for matrix reports.
- `reporting.py`: shared reporting utilities.
- `runtime_evidence.py`: runtime evidence report helpers.
- `scoring_guardrails.py`: trace and scoring guardrail checks.
- `toolchain.py`: ROCm toolchain registry, probes, routing request/report models.
- `trust_summary.py`: trust summary report logic.
- `utils.py`: miscellaneous core helpers.
- `__init__.py`: public core re-export surface.

## Driver Directory

`src/sol_execbench/driver/` contains host staging code and generated templates.

- `problem_packager.py`: `ProblemPackager`, HIP offload target detection, source
  staging, safetensors staging, compile command generation, and evaluation
  command generation.
- `templates/build_ext.py`: copied into staging for native extension builds.
- `templates/eval_driver.py`: copied into staging for benchmark execution.
- `__init__.py`: driver package exports.

The template files are source-controlled Python scripts but execute from a
temporary staging directory, so imports and paths must be robust to that context.

## Scripts

`scripts/` contains command-line tools for dataset, report, and release workflows.

- `scripts/run_dataset.py`: dataset-scale runner and compatibility wrapper around
  `src/sol_execbench/core/dataset/`.
- `scripts/download_solexecbench.py` and `scripts/download_data.sh`: download
  helpers.
- `scripts/inspect_dataset.py`: dataset inspection.
- `scripts/report_amd_bound_sanity.py`: AMD bound sanity report.
- `scripts/report_claim_upgrade.py`: claim upgrade report.
- `scripts/report_consistency.py`: consistency report.
- `scripts/report_evaluation_stability.py`: evaluation stability report.
- `scripts/report_paper_denominator.py`: paper denominator report.
- `scripts/report_parity_gaps.py`: parity gap report.
- `scripts/report_trust_summary.py`: trust summary report.
- `scripts/diff_matrix_reports.py`: matrix diff CLI.
- `scripts/export_matrix_schema.py`: matrix schema export.
- `scripts/check_dataset_redistribution.py`: redistribution policy check.
- `scripts/check_prerelease_readiness.py`: release readiness checker.
- `scripts/build_prerelease_artifact_bundle.py`: release artifact bundle builder.
- `scripts/release_candidate_validation.py`: release candidate validation.
- `scripts/run_docker.sh`: Docker build/run helper.

Scripts use `argparse` or shell conventions and should remain thin wrappers when
logic can be imported from `src/sol_execbench/core/`.

## Examples

`examples/` contains runnable problem directories. Each example generally has
`definition.json`, `workload.jsonl`, a solution JSON, and source files such as
`kernel.py`, `kernel.hip`, `main.cpp`, or `reference.py`.

Current families include:

- `examples/pytorch/linear_backward/`
- `examples/pytorch/gemma3_swiglu/`
- `examples/triton/rmsnorm/`
- `examples/triton/nemotron_rms_norm/`
- `examples/triton/olmo3_post_norm/`
- `examples/hip_cpp/rmsnorm/`
- `examples/hip_cpp/flux_rope/`
- `examples/hipblas/gemm/`
- `examples/miopen/softmax/`
- `examples/ck/gemm/`
- `examples/rocwmma/gemm/`
- Legacy or parity examples under `examples/cudnn/`, `examples/cutlass/`,
  `examples/cutile/`, and `examples/cute_dsl/`.

Examples are excluded from Ruff checks in `pyproject.toml`.

## Tests

`tests/` is organized by concern:

- `tests/sol_execbench/core/data/`: focused schema tests.
- `tests/sol_execbench/core/bench/`: runtime helper tests.
- `tests/sol_execbench/driver/`: staging, build template, and eval driver tests.
- `tests/sol_execbench/test_*.py`: broader unit, report, policy, documentation,
  dataset, scoring, and guardrail tests.
- `tests/sol_execbench/samples/`: sample benchmark problems used by tests.
- `tests/sol_execbench/fixtures/`: structured fixtures, especially SOLAR
  derivation cases.
- `tests/examples/`: example workflow and ROCm CLI path tests.
- `tests/docker/dependencies/`: Docker/ROCm dependency smoke tests.
- `tests/conftest.py`: pytest configuration and marker behavior.
- `tests/sol_execbench_type_helpers.py`: typing helper tests.

Pytest markers are declared in `pyproject.toml`: `cpp`, `requires_rocm`,
`requires_rdna4`, `requires_cdna3`, and legacy `requires_cutile`.

## Documentation

`docs/` contains public and internal documentation:

- General docs: `docs/GETTING-STARTED.md`, `docs/CONFIGURATION.md`,
  `docs/COOKBOOK.md`, `docs/DEVELOPMENT.md`, `docs/RESEARCHER-GUIDE.md`,
  `docs/ARCHITECTURE.md`, and `docs/TESTING.md`.
- Schema docs: `docs/definition.md`, `docs/workload.md`, `docs/solution.md`,
  and `docs/trace.md`.
- ROCm docs: `docs/rocm.md`, `docs/rocm_libraries.md`,
  `docs/rocm_timing.md`, and `docs/rocm_toolchain_routing.md`.
- Evidence/release docs: `docs/CLAIMS.md`, `docs/research_preview.md`,
  `docs/original_parity.md`, `docs/provenance.md`,
  `docs/prerelease_readiness.md`, `docs/prerelease_artifact_bundle.md`,
  `docs/release_candidate_validation.md`, and versioned release closure/notes.
- Demo evidence: `docs/examples/v1_19_evidence/` and
  `docs/examples/v1_20_evidence_quality/`.
- Internal validation notes: `docs/internal/`.
- Release drafts: `docs/releases/`.

## Docker And Data

- `docker/Dockerfile`: ROCm container definition.
- `docker/entrypoint.sh`: container startup/runtime setup.
- `docker/rocm-targets.json`: supported Docker matrix target metadata.
- `data/SOL-ExecBench/`: local downloaded SOL ExecBench benchmark tree.
- `data/flashinfer-trace/`: local FlashInfer trace assets and safetensors blobs.

`data/` contents are local benchmark assets and should not be treated as source
code for package architecture.

## Naming Conventions

- Python modules and functions use `snake_case`.
- Classes and Pydantic models use `PascalCase`.
- Enum members use uppercase names with string values.
- Tests use descriptive `test_*` names, grouped either by package mirror
  directories or by broad report/policy topics.
- JSON problem files use stable names: `definition.json`, `workload.jsonl`,
  `solution.json`, `config.json`, and family-specific `solution_*.json`.
- Example problem directories are grouped by implementation family and operation,
  such as `examples/triton/rmsnorm/` or `examples/hipblas/gemm/`.

## Generated And Local Artifacts

The repository contains local cache or generated paths that should not drive
architecture decisions:

- `.pytest_cache/`
- `.ruff_cache/`
- `__pycache__/`
- `*.pyc`
- downloaded trees under `data/`
- temporary staging directories produced by CLI or dataset runs

Ruff excludes generated/build/cache paths and `examples/`/`data/` via
`pyproject.toml`.
