---
last_mapped_commit: f4de6692ee7468e150c112e0cbdcc8842dd0c709
mapped_at: 2026-06-06
---

# Architecture

## System Pattern

SOL ExecBench ROCm is a Python package that evaluates benchmark problem
definitions and solution implementations on AMD ROCm hardware. The dominant
pattern is a schema-driven benchmarking pipeline:

1. Load problem artifacts from JSON/JSONL files.
2. Validate them through Pydantic models.
3. Stage a self-contained evaluation directory.
4. Optionally compile native HIP/C++ code.
5. Run an isolated evaluation driver as a subprocess.
6. Parse canonical `Trace` JSONL output.
7. Produce optional sidecars for profiling, static kernel evidence,
   environment snapshots, dataset closure, and AMD scoring.

The package entry point is declared in `pyproject.toml`:

- `sol-execbench` -> `sol_execbench.cli:cli`
- `sol-execbench-baseline` -> `sol_execbench.cli.baseline:cli`

The core package is under `src/sol_execbench/`. The public import aggregation
surface is `src/sol_execbench/core/__init__.py`, which re-exports data models,
environment diagnostics, toolchain routing models, and benchmark config types.

## Main Layers

### CLI Layer

`src/sol_execbench/cli/main.py` is the primary user-facing command module. It
contains:

- File loaders for `definition.json`, `workload.jsonl`, `solution.json`, and
  optional `config.json`.
- The main evaluation command exposed by `sol-execbench`.
- Subcommands for public contracts, ROCm environment diagnostics, toolchain
  routing, and dataset migration.
- Sidecar writers for no-trace diagnostics, environment snapshots,
  `rocprofv3` profile metadata, and static kernel evidence.
- Output rendering for Rich tables and JSON trace output.

`src/sol_execbench/cli/baseline.py` is a separate comparison CLI. It loads
candidate and baseline trace JSONL files, calls `src/sol_execbench/core/baseline.py`,
and renders text or JSON comparison output.

### Schema And Contract Layer

`src/sol_execbench/core/data/` contains the benchmark contract models:

- `definition.py` defines `Definition`, symbolic axes, tensor specs, dtype
  metadata, and reference-code validation.
- `workload.py` defines concrete workload inputs, scalar/random/custom inputs,
  safetensors inputs, and tolerance settings.
- `solution.py` defines `Solution`, `BuildSpec`, source files, supported ROCm
  languages, supported AMD hardware targets, and compile option validation.
- `trace.py` defines `Trace`, `Evaluation`, `EvaluationStatus`, correctness,
  performance, and environment result models.
- `contract.py` builds the evaluator contract exposed by the CLI.
- `dtypes.py`, `shapes.py`, `json_utils.py`, and `base_model.py` provide shared
  parsing, validation, and schema helpers.

This layer is the compatibility boundary. CLI, dataset tools, driver templates,
and scoring code all flow through these models rather than passing ad hoc dicts
where validation matters.

### Driver And Staging Layer

`src/sol_execbench/driver/problem_packager.py` owns staging. `ProblemPackager`
writes validated problem artifacts and solution sources into a temporary output
directory. It also:

- Copies or symlinks stageable safetensors inputs.
- Detects whether a solution needs native compilation.
- Injects HIP offload architecture flags based on explicit target hardware or
  local ROCm detection.
- Writes `build_ext.py` and `eval_driver.py` templates into staging.
- Returns subprocess commands that the CLI executes from the staging directory.

`src/sol_execbench/driver/templates/build_ext.py` is the staged native extension
builder. It reads the staged `solution.json`, collects HIP/C++ sources and
compile options, and uses PyTorch extension machinery to build the benchmark
kernel shared object.

`src/sol_execbench/driver/templates/eval_driver.py` is the staged runtime driver.
It redirects non-JSON output to stderr, imports Torch after redirecting stdout,
loads staged models, imports reference and user code, allocates inputs/outputs,
checks reward-hack guardrails, runs correctness and timing, and emits canonical
`Trace` JSONL to the original stdout file descriptor.

### Runtime Benchmark Layer

`src/sol_execbench/core/bench/` contains logic used inside the staged driver and
by host-side orchestration:

- `config/benchmark_config.py` and `config/device_config.py` define benchmark
  settings and device options.
- `io.py` generates or loads workload inputs, allocates destination outputs, and
  handles FlashInfer safetensors environment roots.
- `correctness.py` checks shapes/dtypes and computes numerical error statistics.
- `timing.py` and `timing_policy.py` implement latency measurement policy around
  HIP-backed PyTorch device events.
- `eval_runtime.py` provides staged-driver helpers for loading reference/user
  functions, measuring reference and candidate latency, and emitting trace JSONL.
- `reward_hack.py` detects static and runtime attempts to tamper with benchmark
  integrity.
- `static_kernel_evidence.py` and `static_kernel_status.py` collect source and
  kernel evidence for native/static claims.
- `rocm_profiler.py` integrates optional `rocprofv3` profiling.
- `clock_lock.py`, `stderr.py`, and `utils.py` provide focused runtime helpers.

The runtime layer is intentionally importable from `eval_driver.py`; avoid
introducing dependencies here that require repository-relative paths or stdout
side effects.

### Dataset Layer

`src/sol_execbench/core/dataset/` and `scripts/run_dataset.py` implement
dataset-scale execution and evidence management.

The importable core lives in `src/sol_execbench/core/dataset/`:

- `runner.py` wraps CLI invocation, builds reference/custom solution JSON,
  parses trace JSONL, writes summaries, and builds AMD score reports.
- `run_state.py` discovers problems, selects workloads, maps readiness files,
  maps traces by workload identity, and determines closure status.
- `run_closure.py` writes execution closure records and provenance summaries.
- `migration.py`, `manifest.py`, `layout.py`, `inventory.py`, and `checksums.py`
  handle dataset conversion, layout validation, and manifest integrity.
- `readiness.py`, `ready_subset.py`, `sharding.py`, `low_precision.py`,
  `paper_denominator.py`, `parity_gap.py`, `execution_closure.py`, and
  `evidence_refs.py` support release-quality dataset reporting.

`scripts/run_dataset.py` is the large operational script. It preserves a script
interface while delegating many compatibility exports and reusable helpers back
to `src/sol_execbench/core/dataset/`. It supports single-problem execution,
category selection, workload sharding, ready subsets, existing trace reuse,
timeout overrides, safetensors reference checks, execution closure sidecars, and
post-run AMD scoring artifacts.

### Scoring And Evidence Layer

`src/sol_execbench/core/scoring/` contains ROCm scoring and evidence models:

- `amd_score.py` builds AMD-native suite reports and workload scores.
- `amd_sol.py` and `amd_sol_v2.py` derive AMD SOL bound artifacts.
- `amd_bound_*` modules classify, graph, estimate, and sanity-check AMD bound
  evidence.
- `amd_hardware_models.py` loads AMD hardware model data from
  `src/sol_execbench/data/amd_hardware_models/`.
- `solar_derivation.py` and `solar_derivation_status.py` model derivation
  evidence and status.
- `baseline_artifact.py` handles scoring baseline artifact loading.

Adjacent report and guardrail modules live directly under
`src/sol_execbench/core/`, including `baseline.py`, `claim_upgrade.py`,
`compatibility.py`, `consistency.py`, `dependency_matrix.py`,
`docker_matrix.py`, `environment.py`, `evaluation_stability.py`,
`matrix_diff.py`, `reporting.py`, `runtime_evidence.py`,
`scoring_guardrails.py`, `toolchain.py`, and `trust_summary.py`.

### Operational Scripts And Reports

The `scripts/` directory contains thin CLIs for repository and release workflows.
Most scripts import core modules and render reports, schemas, or validation
artifacts. Examples include:

- `scripts/run_dataset.py` for dataset execution.
- `scripts/download_solexecbench.py` and `scripts/download_data.sh` for data
  acquisition.
- `scripts/export_matrix_schema.py`, `scripts/diff_matrix_reports.py`,
  `scripts/report_consistency.py`, `scripts/report_trust_summary.py`, and
  related report scripts.
- `scripts/build_prerelease_artifact_bundle.py` and
  `scripts/check_prerelease_readiness.py` for release artifacts.

The scripts are operational entry points, while reusable logic should generally
live under `src/sol_execbench/core/`.

## Data Flow

### Single Problem Evaluation

1. A user runs `sol-execbench <problem_dir> --solution <solution-path>` or passes
   explicit `--definition`, `--workload`, and `--solution` paths.
2. `src/sol_execbench/cli/main.py` loads artifacts into `Definition`,
   `Workload`, `Solution`, and `BenchmarkConfig`.
3. `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py` creates a
   staging directory and writes normalized problem files and solution sources.
4. For native ROCm languages (`hip_cpp`, `hipblas`, `miopen`, `ck`, `rocwmma`),
   the CLI runs the packager's compile command, which stages
   `src/sol_execbench/driver/templates/build_ext.py` and builds the extension.
5. The CLI runs the staged `eval_driver.py` subprocess with a bounded timeout and
   FlashInfer safetensors environment.
6. The staged driver loads reference and user functions, executes each workload,
   checks correctness, measures performance, and emits one `Trace` JSON object
   per workload line.
7. The CLI parses `Trace` objects, optionally writes output JSONL, static
   evidence, profiling evidence, environment snapshots, and diagnostic sidecars,
   then renders a table or JSON to the user.

### Dataset Execution

1. `scripts/run_dataset.py` discovers problem directories with `definition.json`
   and `workload.jsonl`.
2. It builds a reference solution from `Definition.reference`, wraps a custom
   solution file, or uses an existing `solution.json`.
3. It optionally shards workload JSONL files and filters by categories, ready
   subsets, readiness metadata, and reuse policy.
4. It invokes `sol-execbench` through helpers in
   `src/sol_execbench/core/dataset/runner.py`.
5. It stores traces, failure logs, summary JSON/Markdown, execution closure
   records, and AMD score evidence under the selected output directory.
6. Reporting scripts read these artifacts to produce release, parity, matrix,
   trust, and claim-upgrade summaries.

## Key Abstractions

- `Definition`: static problem schema, reference implementation, axes, tensors,
  and optional custom input entrypoint.
- `Workload`: concrete axis values and input generation/loading instructions for
  one benchmark case.
- `Solution`: implementation metadata, source files, language, entry point,
  hardware targets, bindings, and compile options.
- `Trace`: canonical result record linking a definition, workload, solution, and
  optional evaluation.
- `BenchmarkConfig`: runtime settings such as warmups, repeats, tolerances, and
  clock-lock behavior.
- `ProblemPackager`: host-side staging coordinator and native compilation command
  producer.
- `EvaluationStatus`: closed set of benchmark outcomes including pass,
  incorrect shape/dtype/numerics, runtime/compile/timeout errors, and reward
  hack detection.
- `ToolchainRoutingReport` and related models in `src/sol_execbench/core/toolchain.py`:
  diagnostic evidence for ROCm tool availability and routing.
- AMD SOL/scoring artifacts in `src/sol_execbench/core/scoring/`: derived
  evidence used for AMD-native performance claims.

## Entry Points

- Package CLI: `src/sol_execbench/cli/main.py`
- Baseline CLI: `src/sol_execbench/cli/baseline.py`
- Dataset runner: `scripts/run_dataset.py`
- Public package exports: `src/sol_execbench/core/__init__.py`
- Staged evaluation script: `src/sol_execbench/driver/templates/eval_driver.py`
- Staged native build script: `src/sol_execbench/driver/templates/build_ext.py`
- Simple score helper: `src/sol_execbench/sol_score.py`
- Docker runtime entry: `docker/entrypoint.sh`
- Docker environment builder: `scripts/run_docker.sh`

## Boundary And Isolation Rules

- User solution sources are written only inside a staging directory created by
  `ProblemPackager`.
- `SourceFile` paths reject absolute paths and `..` traversal in
  `src/sol_execbench/core/data/solution.py`.
- Native compile flags reject response files, external include/library paths,
  runtime linker controls, and CUDA legacy options unless explicitly allowed for
  ROCm system paths.
- The staged `eval_driver.py` redirects stdout to stderr before importing Torch,
  then emits canonical JSONL only through the preserved real stdout.
- Reward-hack checks run before and after importing user code and around workload
  execution.
- Local downloaded benchmark assets belong under `data/` and are not part of
  package source.

## Extension Points

- Add new solution languages by extending `SupportedLanguages` and the packager,
  build template, and eval runtime paths that understand that language.
- Add new hardware targets by extending `SupportedHardware`, AMD hardware model
  JSON under `src/sol_execbench/data/amd_hardware_models/`, and scoring tests.
- Add new report types as importable core modules under `src/sol_execbench/core/`
  with thin script wrappers under `scripts/`.
- Add new dataset evidence sidecars under `src/sol_execbench/core/dataset/` and
  wire them into `scripts/run_dataset.py` only after the reusable logic exists.
- Add new runtime checks inside `src/sol_execbench/core/bench/` only if they are
  safe to import from the staged driver.
