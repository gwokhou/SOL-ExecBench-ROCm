---
last_mapped_commit: dd5731c42d9f3d417acf552b3a02004cd5039df2
last_mapped_date: 2026-06-08
focus: arch
---

# Architecture

## System Shape

SOL ExecBench ROCm is a schema-driven Python benchmark runner for evaluating
GPU kernel solutions on AMD ROCm. The installed package lives under
`src/sol_execbench/`, with CLI entry points declared in `pyproject.toml`:

- `sol-execbench` -> `sol_execbench.cli:cli`
- `sol-execbench-baseline` -> `sol_execbench.cli.baseline:cli`

The main execution path is:

1. Load `definition.json`, `workload.jsonl`, optional `config.json`, and
   `solution.json`.
2. Validate them through Pydantic models in `src/sol_execbench/core/data/`.
3. Stage a self-contained run directory through
   `src/sol_execbench/driver/problem_packager.py`.
4. Compile native HIP/C++ solutions when needed through the staged
   `build_ext.py` template.
5. Run the staged `eval_driver.py` subprocess.
6. Parse canonical `Trace` JSONL from stdout.
7. Optionally write sidecars for environment, profiling, static kernel evidence,
   dataset closure, scoring, and release evidence.

## Main Layers

### CLI Layer

`src/sol_execbench/cli/main.py` is the primary command surface. It owns artifact
loading, user-facing options, staging lifecycle, subprocess execution, Rich table
rendering, trace output writing, and nonfatal sidecars. It also exposes support
commands for evaluator contracts, ROCm diagnostics, toolchain routing, and
dataset migration.

`src/sol_execbench/cli/baseline.py` is a separate trace comparison CLI. It reads
candidate and baseline trace JSONL files, delegates comparison logic to
`src/sol_execbench/core/baseline.py`, and renders text or JSON.

### Contract And Schema Layer

`src/sol_execbench/core/data/` defines the public benchmark contract:

- `definition.py`: problem metadata, tensor specs, symbolic axes, and reference
  implementation metadata.
- `workload.py`: concrete workload rows, scalar/random/custom/safetensors inputs,
  and tolerance settings.
- `solution.py`: solution metadata, supported ROCm language families, build specs,
  source files, bindings, compile options, and target hardware.
- `trace.py`: result records, status enums, correctness, performance, and
  environment payloads.
- `contract.py`: evaluator contract payload generation for CLI exposure.
- `base_model.py`, `dtypes.py`, `shapes.py`, and `json_utils.py`: shared schema
  and parsing utilities.

The package re-export surface in `src/sol_execbench/core/__init__.py` aggregates
these models with environment, toolchain, and benchmark config types.

### Staging And Driver Layer

`src/sol_execbench/driver/problem_packager.py` converts validated in-memory
objects into a staging directory. It writes normalized JSON artifacts, writes
solution sources, exposes stageable safetensors blobs, injects HIP offload
architecture flags, and returns compile/evaluate subprocess commands.

`src/sol_execbench/driver/templates/build_ext.py` is copied into staging for
native solution builds. It reads staged `solution.json`, gathers HIP/C++ sources
and compile options, and builds `benchmark_kernel.so` with PyTorch extension
machinery.

`src/sol_execbench/driver/templates/eval_driver.py` is copied into staging for
runtime evaluation. It redirects library noise to stderr before importing Torch,
loads staged problem artifacts, imports reference and user code, runs integrity
guards, executes workloads, checks correctness, measures latency, and emits one
strict JSON `Trace` object per workload.

### Benchmark Runtime Layer

`src/sol_execbench/core/bench/` contains code used both by the host CLI and the
staged evaluation driver:

- `config/benchmark_config.py` and `config/device_config.py` define benchmark and
  device settings.
- `io.py` generates inputs, loads safetensors, allocates outputs, and builds
  FlashInfer safetensors environment roots.
- `correctness.py` validates output shape/dtype and computes numerical errors.
- `timing.py` and `timing_policy.py` implement HIP-backed timing policy.
- `eval_runtime.py` loads staged functions, measures reference/candidate latency,
  emits trace JSONL, and provides driver helpers.
- `reward_hack.py` blocks static and runtime benchmark tampering attempts.
- `rocm_profiler.py` wraps optional `rocprofv3` collection.
- `static_kernel_evidence.py` and `static_kernel_status.py` collect and classify
  native kernel artifacts.
- `clock_lock.py`, `stderr.py`, and `utils.py` provide focused runtime helpers.

Because `eval_driver.py` imports this layer from a temporary staging context,
runtime modules should avoid repository-relative side effects and stdout output.

### Dataset Layer

Reusable dataset logic lives in `src/sol_execbench/core/dataset/`; the operational
entry point is `scripts/run_dataset.py`.

The dataset layer handles problem discovery, layout inspection, migration,
manifest writing, readiness classification, workload sharding, low-precision
policy, long-tail exclusions, trace reuse, execution closure, paper denominator
reports, parity gap reports, and AMD score evidence.

Key modules include `runner.py`, `run_state.py`, `run_closure.py`,
`execution_closure.py`, `inventory.py`, `layout.py`, `manifest.py`,
`migration.py`, `readiness.py`, `ready_subset.py`, `sharding.py`,
`low_precision.py`, `long_tail_exclusions.py`, `paper_denominator.py`, and
`parity_gap.py`.

### Scoring And Evidence Layer

`src/sol_execbench/core/scoring/` contains AMD-native scoring and SOL evidence:

- `amd_score.py`: workload and suite score report construction.
- `amd_sol.py` and `amd_sol_v2.py`: AMD SOL bound artifacts.
- `amd_bound_classification.py`, `amd_bound_estimate_families.py`,
  `amd_bound_estimates.py`, `amd_bound_graph.py`, and `amd_bound_sanity.py`:
  bound modeling, derivation, graphing, and sanity checks.
- `amd_hardware_models.py`: packaged AMD hardware model loading from
  `src/sol_execbench/data/amd_hardware_models/`.
- `solar_derivation.py` and `solar_derivation_status.py`: SOLAR derivation
  evidence and status modeling.
- `baseline_artifact.py`: scoring baseline artifact loading.

Adjacent evidence/report modules live directly in `src/sol_execbench/core/`,
including `compatibility.py`, `dependency_matrix.py`, `docker_matrix.py`,
`environment.py`, `runtime_evidence.py`, `matrix_diff.py`, `consistency.py`,
`evaluation_stability.py`, `claim_upgrade.py`, `trust_summary.py`,
`reporting.py`, `scoring_guardrails.py`, and `toolchain.py`.

### Script And Release Layer

`scripts/` contains operational CLIs for dataset execution, downloads, report
generation, schema export, Docker helpers, and release readiness. Reusable
business logic generally belongs under `src/sol_execbench/core/`; scripts should
stay as thin command wrappers when practical.

`docs/` contains public docs, schema docs, ROCm docs, evidence examples, internal
validation notes, and release material. Several tests assert doc and evidence
contracts, so docs are part of the repository's verified behavior.

`docker/` provides ROCm container build/runtime support and target matrix
metadata used by Docker and dependency preflight checks.

## Data Flow

### Single Problem Evaluation

1. The user invokes `sol-execbench <problem_dir> --solution <solution-path>` or
   passes explicit artifact paths.
2. `src/sol_execbench/cli/main.py` loads artifacts into `Definition`, `Workload`,
   `Solution`, and `BenchmarkConfig`.
3. `ProblemPackager` writes a staging directory containing normalized problem
   files and solution sources.
4. For native ROCm languages such as `hip_cpp`, `hipblas`, `miopen`, `ck`, and
   `rocwmma`, the CLI runs the packager compile command to produce
   `benchmark_kernel.so`.
5. The CLI runs staged `eval_driver.py` with a bounded timeout and ROCm-related
   environment.
6. The staged driver imports reference and user functions, runs each workload,
   checks correctness and integrity, measures latency, and writes `Trace` JSONL.
7. The CLI parses traces, writes optional artifacts, and renders table or JSON
   output.

### Dataset Execution

1. `scripts/run_dataset.py` discovers dataset problem directories and applies
   category, readiness, sharding, exclusion, and reuse policy.
2. It builds reference/custom/explicit solution JSON through helpers in
   `src/sol_execbench/core/dataset/runner.py`.
3. It invokes `sol-execbench` for selected workloads.
4. It stores traces, logs, summaries, execution closure records, derived evidence,
   and AMD score reports under the chosen output directory.
5. Report scripts read those artifacts to produce compatibility, parity,
   denominator, stability, consistency, trust, and claim-upgrade summaries.

## Extension Points

- Add or revise benchmark artifact schemas in `src/sol_execbench/core/data/` and
  update schema/docs/tests together.
- Add runtime evaluation behavior in `src/sol_execbench/core/bench/` when it must
  run inside staged `eval_driver.py`.
- Add host orchestration behavior in `src/sol_execbench/cli/main.py` or
  `src/sol_execbench/driver/problem_packager.py` when it affects staging,
  subprocesses, or sidecars.
- Add reusable dataset/report logic under `src/sol_execbench/core/dataset/` or
  `src/sol_execbench/core/`, then expose thin scripts in `scripts/`.
- Add AMD scoring or bound evidence under `src/sol_execbench/core/scoring/`.
