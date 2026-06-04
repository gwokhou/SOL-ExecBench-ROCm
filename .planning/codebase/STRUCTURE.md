# Structure

Last updated: 2026-06-04

## Repository Layout

```text
.
├── src/sol_execbench/        Python package source
├── tests/                    Pytest suite and sample problem assets
├── examples/                 Runnable example benchmark problems
├── scripts/                  Dataset, report, release, and Docker helper scripts
├── docs/                     Public and internal documentation
├── docker/                   ROCm Docker environment files
├── data/                     Local benchmark assets and downloaded data
├── .planning/                GSD project state, roadmap, milestones, and codebase maps
├── pyproject.toml            Package, dependency, CLI, pytest, Ruff, ty, and uv config
├── README.md                 Project overview and quickstart
├── CONTRIBUTING.md           Contribution workflow
├── SECURITY.md               Security policy
├── LICENSE                   Apache-2.0 license
└── provenance.toml           Provenance and compliance metadata
```

Generated caches such as `.venv/`, `.pytest_cache/`, `.ruff_cache/`,
`__pycache__/`, build outputs, and downloaded benchmark data are not source
structure.

## Package Layout

```text
src/sol_execbench/
├── __init__.py
├── sol_score.py
├── cli/
│   ├── __init__.py
│   ├── main.py
│   └── baseline.py
├── core/
│   ├── __init__.py
│   ├── bench/
│   ├── data/
│   ├── dataset/
│   ├── scoring/
│   ├── baseline.py
│   ├── claim_upgrade.py
│   ├── compatibility.py
│   ├── consistency.py
│   ├── dependency_matrix.py
│   ├── diagnostics.py
│   ├── docker_matrix.py
│   ├── environment.py
│   ├── evaluation_stability.py
│   ├── matrix_diff.py
│   ├── reporting.py
│   ├── runtime_evidence.py
│   ├── scoring_guardrails.py
│   ├── toolchain.py
│   ├── trust_summary.py
│   └── utils.py
├── data/
│   └── amd_hardware_models/
└── driver/
    ├── __init__.py
    ├── problem_packager.py
    └── templates/
```

## Key Source Locations

### `src/sol_execbench/cli/`

User-facing command-line code.

- `main.py` implements the `sol-execbench` command, evaluator flow, contract,
  doctor, toolchain, and dataset migration commands.
- `baseline.py` implements the `sol-execbench-baseline` command.
- `__init__.py` exposes the main CLI object used by the package script entry
  point.

### `src/sol_execbench/core/data/`

Typed benchmark schemas and public contracts.

- `definition.py` contains `Definition`, axis models, tensor specs, dtype
  support, reference-code validation, and resolved-axis helpers.
- `workload.py` contains workload rows, input specs, tolerance metadata, and
  workload validation.
- `solution.py` contains source files, supported ROCm language categories,
  supported hardware, build specs, bindings, compile options, entry-point
  validation, and unsafe native flag rejection.
- `trace.py` contains canonical trace output models and evaluation status.
- `contract.py` builds evaluator compatibility metadata.
- `dtypes.py`, `shapes.py`, and `json_utils.py` provide support helpers.
- `base_model.py` centralizes common Pydantic behavior and constrained types.

### `src/sol_execbench/core/bench/`

Runtime helpers used by the generated evaluation driver and CLI.

- `config/benchmark_config.py` and `config/device_config.py` define benchmark
  and device config.
- `io.py` creates inputs, allocates outputs, and loads safetensors data.
- `correctness.py` handles seed control, output shape/dtype checks, and error
  metrics.
- `timing.py` and `timing_policy.py` define timing primitives and policy.
- `clock_lock.py` checks GPU clock-lock status.
- `reward_hack.py` contains static and runtime reward-hack detection.
- `eval_runtime.py` provides helpers imported by `driver/templates/eval_driver.py`.
- `rocm_profiler.py` wraps optional `rocprofv3` profiling collection.
- `static_kernel_evidence.py` and `static_kernel_status.py` collect and
  classify native static kernel evidence.
- `utils.py` contains runtime utility functions for evaluation assembly.

### `src/sol_execbench/core/dataset/`

Dataset layout, migration, execution, closure, and readiness logic.

Important modules include:

- `layout.py`, `manifest.py`, `checksums.py`, and `inventory.py` for local
  dataset structure and metadata.
- `migration.py` for SOL ExecBench and FlashInfer trace migration.
- `readiness.py`, `ready_subset.py`, `categories.py`, and `low_precision.py`
  for classifying runnable and compatible subsets.
- `runner.py` for importable dataset-scale CLI invocation, solution wrapping,
  trace parsing, timing evidence, and AMD-native report assembly.
- `run_state.py`, `run_closure.py`, `execution_closure.py`, and
  `evidence_refs.py` for reuse, closure, and evidence-reference tracking.
- `paper_denominator.py`, `parity_gap.py`, and `sharding.py` for denominator
  accounting, parity reports, and deterministic shard plans.

### `src/sol_execbench/core/scoring/`

AMD-native scoring and SOL/SOLAR evidence.

- `amd_score.py` builds guarded per-workload and suite-level AMD-native score
  reports.
- `amd_sol.py` and `amd_sol_v2.py` build AMD SOL bound artifacts.
- `amd_bound_*.py` modules implement bound classifications, estimates,
  families, graphs, and sanity checks.
- `amd_hardware_models.py` loads packaged hardware model metadata.
- `baseline_artifact.py` models scoring baselines.
- `solar_derivation.py` and `solar_derivation_status.py` model SOLAR-derived
  evidence and aggregate status.

### `src/sol_execbench/core/`

Cross-cutting package logic.

- `environment.py` captures local ROCm/PyTorch/GPU/tool diagnostics.
- `toolchain.py` models ROCm evidence-tool capabilities and routing decisions.
- `compatibility.py` models compatibility matrix evidence and execution
  boundaries.
- `docker_matrix.py` and `dependency_matrix.py` model Docker target and
  dependency evidence.
- `reporting.py` summarizes canonical traces into derived reports.
- `baseline.py` compares new traces against baseline traces.
- `claim_upgrade.py`, `consistency.py`, `evaluation_stability.py`,
  `matrix_diff.py`, `runtime_evidence.py`, `scoring_guardrails.py`, and
  `trust_summary.py` support release-quality evidence and claim checks.

### `src/sol_execbench/driver/`

Staging and generated subprocess files.

- `problem_packager.py` writes staged input/source/config files, injects native
  offload architecture flags, returns compile and execute commands, and parses
  trace JSONL from stdout.
- `templates/build_ext.py` is copied into native staging directories and uses
  `torch.utils.cpp_extension.load` to build `benchmark_kernel.so`.
- `templates/eval_driver.py` is copied into staging directories and performs
  the actual per-workload evaluation.

### `src/sol_execbench/data/`

Packaged data imported by the Python package. Current hardware model data lives
under `src/sol_execbench/data/amd_hardware_models/`.

## Tests

```text
tests/
├── conftest.py
├── sol_execbench_type_helpers.py
├── sol_execbench/
│   ├── core/
│   ├── driver/
│   ├── fixtures/
│   ├── samples/
│   └── test_*.py
├── examples/
├── docker/
│   └── dependencies/
└── samples/
```

Test organization mirrors the package where practical:

- `tests/sol_execbench/core/` covers lower-level bench and data helpers.
- `tests/sol_execbench/driver/` covers staging, native build templates, and
  generated evaluation driver behavior.
- `tests/sol_execbench/test_*.py` covers higher-level dataset, scoring, ROCm,
  docs, release, compatibility, and guardrail behavior.
- `tests/examples/` validates runnable examples and CLI paths.
- `tests/docker/dependencies/` validates Docker/runtime dependencies.
- `tests/sol_execbench/samples/` and `tests/samples/` provide fixture solutions,
  including reward-hack and error-path examples.

Markers are defined in `pyproject.toml`: `cpp`, `requires_rocm`,
`requires_rdna4`, `requires_cdna3`, and legacy `requires_cutile`.

## Examples

```text
examples/
├── pytorch/
├── triton/
├── hip_cpp/
├── hipblas/
├── miopen/
├── ck/
├── rocwmma/
├── cudnn/
├── cutlass/
├── cute_dsl/
└── cutile/
```

Each problem directory generally contains:

- `definition.json`
- `workload.jsonl`
- `reference.py` when a standalone reference file is needed
- `kernel.py`, `kernel.hip`, `main.cpp`, or headers depending on the category
- `solution_*.json`

Active ROCm examples cover PyTorch, Triton ROCm, HIP C++, hipBLAS, MIOpen,
Composable Kernel, and rocWMMA. Legacy CUDA/NVIDIA-named directories remain as
migration or compatibility fixtures and should not be treated as active dual
backend support.

## Scripts

Top-level helper scripts live in `scripts/`.

Dataset and execution scripts:

- `download_data.sh`
- `download_solexecbench.py`
- `inspect_dataset.py`
- `run_dataset.py`

Report and evidence scripts:

- `report_amd_bound_sanity.py`
- `report_claim_upgrade.py`
- `report_consistency.py`
- `report_evaluation_stability.py`
- `report_paper_denominator.py`
- `report_parity_gaps.py`
- `report_trust_summary.py`
- `diff_matrix_reports.py`
- `export_matrix_schema.py`

Release and environment scripts:

- `build_prerelease_artifact_bundle.py`
- `check_dataset_redistribution.py`
- `check_prerelease_readiness.py`
- `release_candidate_validation.py`
- `run_docker.sh`

Scripts are importers of package logic where possible, not the canonical home
for schema or scoring contracts.

## Documentation

`docs/` contains public user documentation, schema references, internal
validation notes, release materials, and research evidence. High-traffic files
include:

- `docs/ARCHITECTURE.md`
- `docs/GETTING-STARTED.md`
- `docs/CONFIGURATION.md`
- `docs/TESTING.md`
- `docs/RESEARCHER-GUIDE.md`
- `docs/rocm.md`
- `docs/rocm_timing.md`
- `docs/rocm_libraries.md`
- `docs/rocm_toolchain_routing.md`
- `docs/static_kernel_evidence.md`
- `docs/provenance.md`
- `docs/compliance.md`

Internal and release-oriented material is grouped under `docs/internal/`,
`docs/releases/`, and `docs/examples/`.

## Docker

`docker/` contains:

- `Dockerfile` for the ROCm evaluation image.
- `entrypoint.sh` for container startup behavior.
- `rocm-targets.json` for target matrix/preflight data.

`scripts/run_docker.sh` is the main wrapper for building and entering the Docker
environment.

## Planning State

`.planning/` contains GSD project state, milestones, phase plans, audits,
notes, research, and generated codebase maps. The current requested codebase
maps live in `.planning/codebase/`.

Important planning files:

- `.planning/PROJECT.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `.planning/MILESTONES.md`
- `.planning/milestones/`
- `.planning/codebase/`

## Naming Conventions

- Python modules, functions, variables, and test files use `snake_case`.
- Classes, dataclasses, Pydantic models, and enums use `PascalCase`.
- Enum members use uppercase names with string values where exposed through
  JSON or CLI contracts.
- CLI commands and options use kebab-case.
- Canonical benchmark input files use fixed names:
  `definition.json`, `workload.jsonl`, `solution.json`, and optional
  `config.json`.
- Example solution files use descriptive names such as `solution_triton.json`,
  `solution_hip.json`, or `solution_python.json`.
- Dataset/report JSON schemas include explicit `schema_version` fields.
- Tests use descriptive `test_*` names and are colocated with related package
  boundaries when possible.

## File Placement Rules

- Add evaluator CLI changes under `src/sol_execbench/cli/`.
- Add public schema changes under `src/sol_execbench/core/data/`.
- Add generated-driver runtime helpers under `src/sol_execbench/core/bench/`.
- Add staging or template changes under `src/sol_execbench/driver/`.
- Add dataset layout, migration, readiness, closure, or sharding logic under
  `src/sol_execbench/core/dataset/`.
- Add AMD scoring, bounds, or SOLAR evidence under
  `src/sol_execbench/core/scoring/`.
- Add importable report helpers under `src/sol_execbench/core/`; add thin
  command wrappers under `scripts/`.
- Add package data under `src/sol_execbench/data/`.
- Add local downloaded benchmark assets under `data/`, not under package
  source.
- Add tests under `tests/sol_execbench/`, `tests/examples/`, or
  `tests/docker/` according to the affected surface.
