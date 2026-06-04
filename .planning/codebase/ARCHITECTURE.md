# Architecture

Last updated: 2026-06-04

## Pattern

SOL ExecBench ROCm is a layered Python CLI package for evaluating GPU kernel
solutions on AMD ROCm hardware. The dominant pattern is:

1. Validate benchmark inputs with typed Pydantic contracts.
2. Stage the problem, workload, solution sources, and generated driver files in
   a temporary execution directory.
3. Compile native ROCm sources when required.
4. Run evaluation in a subprocess that emits canonical `Trace` JSONL.
5. Build optional diagnostic, dataset, scoring, and release-readiness evidence
   around the trace output without mutating the trace schema.

The design keeps user solution execution out of the CLI process. It is an
execution boundary and guardrail layer, not a full security sandbox for
multi-tenant adversarial code.

## Layers

### CLI Layer

Location: `src/sol_execbench/cli/`

The primary entry point is `sol-execbench`, mapped in `pyproject.toml` to
`sol_execbench.cli:cli`. `src/sol_execbench/cli/main.py` owns the user-facing
Click command, problem path resolution, JSON loading, staging lifecycle,
subprocess orchestration, optional evidence sidecars, Rich summaries, and exit
status.

The secondary entry point is `sol-execbench-baseline`, mapped to
`sol_execbench.cli.baseline:cli`, for comparing trace files against baseline
artifacts.

The root CLI also exposes GPU-free metadata and local utility commands:

- `contract` for evaluator compatibility contracts.
- `doctor` for ROCm/PyTorch/tool environment diagnostics.
- `toolchain` for ROCm evidence-tool routing reports.
- `dataset migrate-sol` and `dataset migrate-flashinfer` for local dataset
  migration.

### Contract And Model Layer

Location: `src/sol_execbench/core/data/` and
`src/sol_execbench/core/bench/config/`

This layer defines the public benchmark contracts:

- `Definition` describes symbolic axes, tensors, reference code, custom input
  entry points, and workload semantics.
- `Workload` describes one workload row and its generated, scalar, custom, or
  safetensors-backed inputs.
- `Solution` and `BuildSpec` describe candidate source files, entry points,
  language categories, target AMD hardware, destination-passing behavior,
  bindings, and compile options.
- `Trace`, `Evaluation`, `Correctness`, `Performance`, and `Environment`
  describe canonical per-workload outputs.
- `BenchmarkConfig` carries warmup, repetition, seed, reference timing, and
  clock-lock settings.

These models are reused by the CLI, generated drivers, scripts, tests, dataset
helpers, and reporting modules.

### Driver And Staging Layer

Location: `src/sol_execbench/driver/`

`ProblemPackager` is the central staging abstraction. It writes normalized
`definition.json`, `workload.jsonl`, `solution.json`, `config.json`, and source
files to a temporary directory. For native ROCm categories it can copy
`driver/templates/build_ext.py`, inject HIP offload architecture flags when no
explicit architecture flag is present, and return the native build command.

`ProblemPackager.execute()` copies `driver/templates/eval_driver.py` and returns
the command used by the CLI to run the generated evaluation process. The
packager removes the staging directory unless `--keep-staging` is set.

### Benchmark Runtime Layer

Location: `src/sol_execbench/core/bench/` plus
`src/sol_execbench/driver/templates/eval_driver.py`

The generated evaluation driver is a self-contained script copied into the
staging directory. It redirects non-JSON stdout noise to stderr, imports PyTorch,
loads typed staged inputs, imports the reference implementation and user
function, performs source and runtime reward-hack checks, generates inputs,
allocates outputs, checks shape and dtype, computes numerical error, measures
latency, and emits one strict JSON `Trace` per workload to the original stdout.

Reusable runtime helpers live in `core/bench/`:

- `io.py` generates and loads inputs.
- `correctness.py` handles deterministic seeds, shape/dtype checks, and error
  statistics.
- `timing.py` and `timing_policy.py` handle HIP-backed PyTorch event timing and
  timing policy decisions.
- `reward_hack.py` contains static and runtime guardrails.
- `rocm_profiler.py` wraps optional `rocprofv3` evidence collection.
- `static_kernel_evidence.py` and `static_kernel_status.py` collect and classify
  native static evidence.
- `eval_runtime.py` provides importable helpers used by the generated driver.

### Evidence And Reporting Layer

Location: `src/sol_execbench/core/`

This layer keeps diagnostic and derived evidence separate from canonical traces:

- `environment.py` captures ROCm, PyTorch, GPU, visible-device, and tool
  diagnostics.
- `toolchain.py` selects ROCm tools for requested evidence levels and artifact
  types.
- `compatibility.py`, `docker_matrix.py`, and `dependency_matrix.py` model
  compatibility, Docker target, and dependency evidence.
- `reporting.py` summarizes traces and derived diagnostics.
- `baseline.py` compares trace runs against baselines.
- `claim_upgrade.py`, `consistency.py`, `evaluation_stability.py`,
  `matrix_diff.py`, `runtime_evidence.py`, `scoring_guardrails.py`, and
  `trust_summary.py` build release and claim-boundary reports.

Optional sidecars, including environment snapshots, rocprofv3 metadata, static
kernel evidence, and no-trace diagnostics, are diagnostic artifacts. They do not
alter the canonical trace JSONL contract.

### Dataset Layer

Location: `src/sol_execbench/core/dataset/`

Dataset helpers support local migration, layout discovery, inventory,
checksums, readiness classification, ready subsets, paper denominator
accounting, parity gaps, low-precision compatibility, run state, execution
closure, evidence refs, and deterministic sharding.

The dataset layer is used by CLI dataset commands and by scripts such as
`scripts/download_solexecbench.py`, `scripts/inspect_dataset.py`, and
`scripts/run_dataset.py`.

### Scoring Layer

Location: `src/sol_execbench/core/scoring/` and `src/sol_execbench/sol_score.py`

The scoring layer provides AMD-native derived scoring and SOL/SOLAR evidence:

- AMD hardware models and bound estimates.
- Bound graph and bound artifact construction.
- AMD SOL v1/v2 helpers.
- SOLAR derivation status and evidence.
- Baseline artifact support.
- Guarded AMD-native score reports with explicit claim-level warnings.

Packaged hardware model data lives under `src/sol_execbench/data/`.

### Operational Layer

Locations: `scripts/`, `docker/`, `examples/`, `docs/`, `tests/`

Scripts provide dataset download, dataset execution, matrix/report generation,
release checks, trust summaries, and prerelease bundle assembly. Docker files
define the ROCm evaluation environment and dependency preflight surface.
Examples provide runnable problem layouts for supported and legacy categories.
Docs describe public contracts, ROCm behavior, timing, validation, release, and
research guidance. Tests cover schemas, drivers, examples, dataset flows,
Docker checks, scoring, evidence reports, and guardrails.

## Data Flow

### Single Problem Evaluation

1. User invokes `sol-execbench <problem_dir> --solution <solution.json>` or
   provides explicit `--definition`, `--workload`, and `--solution` paths.
2. The CLI loads JSON into `Definition`, `Workload`, `Solution`, and
   `BenchmarkConfig`.
3. The CLI creates a `sol_execbench_*` temporary staging directory and
   constructs a `ProblemPackager`.
4. The packager writes normalized staged JSON files and source files.
5. If the solution is native ROCm (`hip_cpp`, `hipblas`, `miopen`, `ck`, or
   `rocwmma`), the CLI runs the generated `build_ext.py` subprocess to produce
   `benchmark_kernel.so`.
6. The CLI copies and runs the generated `eval_driver.py` subprocess with the
   staged problem.
7. The driver evaluates each workload and emits strict `Trace` JSONL to stdout.
8. The CLI parses stdout into `Trace` models, writes optional output and
   sidecars, prints either JSON or a Rich table, cleans staging unless requested,
   and exits with status 0 only if all workloads passed.

### Dataset-Scale Evaluation

1. Dataset scripts inspect or migrate benchmark assets into local layouts.
2. `core/dataset/runner.py` builds reference or custom solution JSON for each
   problem.
3. The runner invokes the installed `sol-execbench` CLI as a subprocess for
   each selected job.
4. Trace JSONL, CLI logs, timing evidence, derived evidence refs, AMD SOL bound
   artifacts, and AMD-native score reports are written into script-managed
   output directories.
5. Dataset closure and reuse helpers classify stale, missing, failed, reused,
   and selected workload evidence.

## Entry Points

| Entry point | Location | Purpose |
| --- | --- | --- |
| `sol-execbench` | `src/sol_execbench/cli/main.py` | Main evaluator and metadata CLI. |
| `sol-execbench-baseline` | `src/sol_execbench/cli/baseline.py` | Baseline comparison CLI. |
| `scripts/run_dataset.py` | `scripts/run_dataset.py` | Batch execution over downloaded benchmark assets. |
| `scripts/download_solexecbench.py` | `scripts/download_solexecbench.py` | Download and prepare benchmark source assets. |
| `scripts/inspect_dataset.py` | `scripts/inspect_dataset.py` | Inspect migrated/local dataset state. |
| `scripts/report_*.py` | `scripts/` | Generate scoring, parity, consistency, trust, and evidence reports. |
| `scripts/run_docker.sh` | `scripts/run_docker.sh` | Build and enter ROCm Docker evaluation environment. |
| `docker/entrypoint.sh` | `docker/entrypoint.sh` | Container startup and runtime setup. |

## Key Abstractions

| Abstraction | Location | Role |
| --- | --- | --- |
| `ProblemPackager` | `src/sol_execbench/driver/problem_packager.py` | Stages inputs, sources, generated files, build commands, and execution commands. |
| `Definition` | `src/sol_execbench/core/data/definition.py` | Benchmark operation contract and reference implementation. |
| `Workload` | `src/sol_execbench/core/data/workload.py` | Per-workload axes, input values, tolerances, and input loading metadata. |
| `Solution` / `BuildSpec` | `src/sol_execbench/core/data/solution.py` | Candidate implementation contract, source files, entry point, languages, hardware, and compile options. |
| `BenchmarkConfig` | `src/sol_execbench/core/bench/config/benchmark_config.py` | Runtime configuration for timing, seeds, clock locks, and references. |
| `Trace` | `src/sol_execbench/core/data/trace.py` | Canonical benchmark output record. |
| `EnvironmentSnapshot` / `EnvironmentDiagnostics` | `src/sol_execbench/core/environment.py` | Optional environment evidence and diagnostics. |
| `ToolchainRoutingReport` | `src/sol_execbench/core/toolchain.py` | ROCm tool routing and capability evidence. |
| `StaticKernelEvidenceSidecar` | `src/sol_execbench/core/bench/static_kernel_evidence.py` | Diagnostic native artifact and extractor evidence. |
| `TraceRunSummary` / `DerivedEvidenceReport` | `src/sol_execbench/core/reporting.py` | Derived summaries around canonical traces. |
| `AmdNativeScore` / `AmdNativeSuiteReport` | `src/sol_execbench/core/scoring/amd_score.py` | Guarded AMD-native scoring reports. |

## ROCm Solution Boundary

The active ROCm solution categories are:

- `pytorch`
- `triton`
- `hip_cpp`
- `hipblas`
- `miopen`
- `ck`
- `rocwmma`

Native categories share the HIP/C++ extension build path. Legacy CUDA/NVIDIA
schema values are rejected with ROCm migration guidance. Native compile flags
are constrained to prevent response files, arbitrary host path injection, and
runtime loader path manipulation. Python/Triton and native categories are kept
separate in `Solution` validation.

## Architectural Constraints

- The supported baseline is Python 3.12+ and ROCm 7.x.
- Public benchmark schemas and trace JSONL semantics are intentionally stable.
- CUDA/NVIDIA paths are treated as legacy migration residue, not as a dual
  backend.
- Canonical benchmark authority remains trace JSONL; reports and sidecars are
  derived or diagnostic evidence unless explicitly documented otherwise.
- GPU execution depends on ROCm-capable AMD hardware, PyTorch ROCm, and access
  to GPU device nodes in host or container environments.
