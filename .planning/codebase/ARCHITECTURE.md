<!-- refreshed: 2026-06-01 -->
# Architecture

**Analysis Date:** 2026-06-01

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                   User-Facing Command Layer                  │
├──────────────────┬──────────────────┬───────────────────────┤
│ Evaluator CLI    │ Baseline CLI     │ Dataset/report scripts │
│ `src/sol_execbench/cli/main.py` │ `src/sol_execbench/cli/baseline.py` │ `scripts/` │
└────────┬─────────┴────────┬─────────┴──────────┬────────────┘
         │                  │                     │
         ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Typed Contracts And Core Services            │
│ `src/sol_execbench/core/data/`, `core/dataset/`, `core/scoring/`, `core/bench/` │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Staged Execution Boundary                   │
│ `src/sol_execbench/driver/problem_packager.py`               │
│ `src/sol_execbench/driver/templates/eval_driver.py`          │
│ `src/sol_execbench/driver/templates/build_ext.py`            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         ROCm Runtime, Native Build Artifacts, JSONL Traces   │
│ `/tmp/sol_execbench_*`, `benchmark_kernel.so`, trace sidecars │
└─────────────────────────────────────────────────────────────┘
```

SOL ExecBench is a layered Python CLI package. The normal evaluation path loads typed Pydantic contracts, stages a problem into an isolated temporary directory, optionally compiles HIP/C++ sources, runs a generated evaluation driver in a subprocess, parses JSONL `Trace` rows, and writes optional diagnostic sidecars. Dataset and reporting scripts reuse the same core contracts instead of reimplementing parsing or scoring logic.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `SolExecbenchCli` | Dispatches root evaluation calls plus `contract`, `doctor`, and `toolchain` metadata subcommands. | `src/sol_execbench/cli/main.py` |
| `_evaluate_cli` | Loads problem inputs, creates staging, runs compile/evaluation subprocesses, parses traces, writes sidecars, sets exit code. | `src/sol_execbench/cli/main.py` |
| `ProblemPackager` | Writes normalized staged files, solution sources, generated templates, native build commands, and trace parser. | `src/sol_execbench/driver/problem_packager.py` |
| `eval_driver.py` template | Runs user code in the staging subprocess, performs correctness/timing/reward-hack checks, emits strict JSONL traces. | `src/sol_execbench/driver/templates/eval_driver.py` |
| `build_ext.py` template | Builds native ROCm sources through `torch.utils.cpp_extension.load` and produces `benchmark_kernel.so`. | `src/sol_execbench/driver/templates/build_ext.py` |
| Data contracts | Own benchmark schemas for definitions, workloads, solutions, evaluator contract, and traces. | `src/sol_execbench/core/data/` |
| Benchmark helpers | Own runtime loading, input generation, output checks, timing, profiling, clock locks, and static kernel evidence. | `src/sol_execbench/core/bench/` |
| Dataset services | Own dataset discovery, inventory, manifests, readiness, execution closure, parity gaps, and paper denominator reports. | `src/sol_execbench/core/dataset/` |
| Scoring services | Own AMD hardware models, bound graph extraction, derived SOL artifacts, AMD-native scores, and score guardrails. | `src/sol_execbench/core/scoring/` |
| Environment/toolchain services | Own ROCm probes, compatibility matrices, Docker target selection, dependency policy, and runtime evidence. | `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/compatibility.py`, `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/runtime_evidence.py` |
| Reporting services | Own baseline comparison, trace summaries, consistency/stability/claim/trust reports, and matrix diffs. | `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/consistency.py`, `src/sol_execbench/core/evaluation_stability.py`, `src/sol_execbench/core/claim_upgrade.py`, `src/sol_execbench/core/trust_summary.py`, `src/sol_execbench/core/matrix_diff.py` |
| Batch runner | Discovers dataset problems, wraps reference/custom solutions, invokes CLI subprocesses, writes trace and derived score evidence. | `scripts/run_dataset.py` |

## Pattern Overview

**Overall:** Layered CLI plus isolated staged evaluator subprocess.

**Key Characteristics:**
- Use Pydantic models in `src/sol_execbench/core/data/` and related report modules as the authority for external JSON schemas.
- Keep user solution execution outside the CLI process through `ProblemPackager` and generated templates in `src/sol_execbench/driver/templates/`.
- Keep benchmark execution, dataset reporting, compatibility evidence, and scoring in separate core subpackages.
- Treat optional evidence (`rocprofv3`, static kernel artifacts, environment snapshots) as sidecars that do not change trace correctness authority.
- Use `src/sol_execbench/core/__init__.py` as the public convenience export for common evaluator contracts, but import specialized services from their concrete modules.

## Layers

**CLI Layer:**
- Purpose: Parse command-line arguments, load files, call core services, run subprocesses, render user output.
- Location: `src/sol_execbench/cli/`
- Contains: `main.py` for evaluation and metadata subcommands; `baseline.py` for baseline comparison.
- Depends on: `src/sol_execbench/core/`, `src/sol_execbench/driver/`, Click, Rich, `subprocess`.
- Used by: `pyproject.toml` console scripts, `scripts/run_dataset.py`, tests under `tests/sol_execbench/`.

**Contract Layer:**
- Purpose: Validate and serialize benchmark definitions, workloads, solutions, trace rows, evaluator contract, and configuration.
- Location: `src/sol_execbench/core/data/` and `src/sol_execbench/core/bench/config/`
- Contains: `Definition`, `Workload`, `Solution`, `BuildSpec`, `Trace`, `EvaluationStatus`, `BenchmarkConfig`.
- Depends on: Pydantic, standard library, limited Torch dtype adapters in `src/sol_execbench/core/data/dtypes.py`.
- Used by: CLI, driver templates, dataset tools, scoring tools, tests, docs.

**Benchmark Runtime Layer:**
- Purpose: Execute benchmark semantics inside the staged process.
- Location: `src/sol_execbench/core/bench/`
- Contains: input loading/generation, correctness checks, timing helpers, runtime import helpers, reward-hack checks, clock lock helpers, profiler and static-evidence collectors.
- Depends on: PyTorch ROCm, safetensors, ROCm command-line tools, typed contracts.
- Used by: `src/sol_execbench/driver/templates/eval_driver.py`, CLI evidence sidecars, dataset runner timing evidence.

**Execution Boundary Layer:**
- Purpose: Stage untrusted solution inputs and generated driver files into a temporary directory and return subprocess commands.
- Location: `src/sol_execbench/driver/`
- Contains: `ProblemPackager`, `eval_driver.py`, `build_ext.py`.
- Depends on: core contracts, filesystem, subprocess, `torch.utils.cpp_extension` in the native build template.
- Used by: evaluator CLI and direct packager tests.

**Dataset Layer:**
- Purpose: Analyze and prepare downloaded SOL ExecBench datasets without embedding dataset policy in the evaluator.
- Location: `src/sol_execbench/core/dataset/`
- Contains: category validation, checksums, layout, inventory, readiness, manifest, ready subsets, run state, execution closure, parity gap, paper denominator helpers.
- Depends on: Pydantic, file system JSON/JSONL, core data contracts.
- Used by: `scripts/inspect_dataset.py`, `scripts/run_dataset.py`, report scripts, tests.

**Scoring Layer:**
- Purpose: Derive AMD-native scores and guarded SOL-bound artifacts from traces, definitions, workloads, and hardware models.
- Location: `src/sol_execbench/core/scoring/`
- Contains: AMD hardware model loading, bound graph extraction, operator estimates, SOL v1/v2 artifacts, solar derivation, AMD score reports, bound sanity checks.
- Depends on: core data contracts, packaged JSON in `src/sol_execbench/data/amd_hardware_models/`, optional Torch FX analysis.
- Used by: `scripts/run_dataset.py`, `scripts/report_amd_bound_sanity.py`, score tests.

**Evidence And Compatibility Layer:**
- Purpose: Capture ROCm environment, dependency, Docker, compatibility-matrix, toolchain-routing, and runtime-evidence state.
- Location: `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/compatibility.py`, `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/runtime_evidence.py`
- Contains: Pydantic report models, probe runners, target selection, schema export, preflight classifiers.
- Depends on: subprocess probes and JSON manifests such as `docker/rocm-targets.json`.
- Used by: CLI `doctor` and `toolchain`, Docker scripts, report scripts, CI-style tests.

## Data Flow

### Primary Request Path

1. Console script `sol-execbench` points to `sol_execbench.cli:cli` in `pyproject.toml`.
2. `SolExecbenchCli.main()` dispatches subcommands or evaluation (`src/sol_execbench/cli/main.py:847`).
3. `_evaluate_cli()` resolves positional problem directories or explicit `--definition`, `--workload`, `--solution`, `--config` paths (`src/sol_execbench/cli/main.py:590`).
4. `_load_definition()`, `_load_workloads()`, `_load_solution()`, and `_load_config()` instantiate `Definition`, `Workload`, `Solution`, and `BenchmarkConfig` (`src/sol_execbench/cli/main.py:90`).
5. `_evaluate_cli()` creates `tempfile.mkdtemp(prefix="sol_execbench_")` and constructs `ProblemPackager` (`src/sol_execbench/cli/main.py:630`).
6. `ProblemPackager.__init__()` writes `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, and source files into the staging directory (`src/sol_execbench/driver/problem_packager.py:91`).
7. For native ROCm solutions, `_evaluate_cli()` calls `ProblemPackager.compile()` and runs `python build_ext.py` in the staging directory (`src/sol_execbench/cli/main.py:644`).
8. `build_ext.py` calls `torch.utils.cpp_extension.load()` and normalizes the output artifact to `benchmark_kernel.so` (`src/sol_execbench/driver/templates/build_ext.py:38`).
9. `ProblemPackager.execute()` writes `eval_driver.py` and returns `python eval_driver.py` (`src/sol_execbench/driver/problem_packager.py:217`).
10. `_evaluate_cli()` runs the evaluation command directly or through `collect_rocprofv3_profile()` (`src/sol_execbench/cli/main.py:693`).
11. `eval_driver.py` loads staged problem files, imports reference and user functions, reviews source, checks integrity, executes correctness rounds, times the solution, and emits one JSON `Trace` per workload (`src/sol_execbench/driver/templates/eval_driver.py:92`).
12. `ProblemPackager.convert_stdout_to_traces()` parses JSONL stdout into `Trace` models (`src/sol_execbench/driver/problem_packager.py:242`).
13. `_evaluate_cli()` writes trace JSONL and optional environment/profile/static-evidence sidecars, prints JSON or a Rich table, cleans staging unless requested, and exits nonzero unless all traces passed (`src/sol_execbench/cli/main.py:739`).

### Dataset Batch Flow

1. `scripts/run_dataset.py` discovers problem directories by delegating to `core.dataset.run_state.discover_problems()` (`scripts/run_dataset.py:107`).
2. The script builds a solution JSON from a provided solution file, custom Python file, or `Definition.reference` (`scripts/run_dataset.py:159`).
3. It builds a `sol-execbench --definition --workload --solution --json` command (`scripts/run_dataset.py:246`).
4. It invokes the CLI in a subprocess and parses JSON trace lines from stdout (`scripts/run_dataset.py:279`).
5. It can build timing sidecars with `collect_source_timing_evidence()` (`scripts/run_dataset.py:396`).
6. It derives AMD-native scores, SOL bound artifacts, and solar derivation sidecars through `src/sol_execbench/core/scoring/` (`scripts/run_dataset.py:530`).
7. It writes dataset execution closure and evidence references through `src/sol_execbench/core/dataset/run_closure.py`.

### Metadata And Evidence Flow

1. `sol-execbench contract --json` returns `build_evaluator_contract()` from `src/sol_execbench/core/data/contract.py` (`src/sol_execbench/cli/main.py:776`).
2. `sol-execbench doctor --json` returns `build_environment_diagnostics()` from `src/sol_execbench/core/environment.py` (`src/sol_execbench/cli/main.py:787`).
3. `sol-execbench toolchain --json` returns `build_toolchain_routing_report()` or `default_toolchain_registry()` from `src/sol_execbench/core/toolchain.py` (`src/sol_execbench/cli/main.py:798`).
4. Docker and runtime evidence commands use the models and classifiers in `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, and `src/sol_execbench/core/runtime_evidence.py`.

**State Management:**
- Runtime benchmark state is scoped to the generated `eval_driver.py` process and its staging directory.
- Pydantic models are immutable where behavior depends on hashing, such as `Solution` (`src/sol_execbench/core/data/solution.py:262`).
- The CLI keeps small module-level constants for environment-variable names and option values in `src/sol_execbench/cli/main.py:82`.
- Dataset/scoring/reporting modules return explicit model instances or dictionaries; they do not maintain long-lived in-memory registries.

## Key Abstractions

**Problem Contracts:**
- Purpose: Represent externally visible benchmark inputs and outputs.
- Examples: `Definition` in `src/sol_execbench/core/data/definition.py`, `Workload` in `src/sol_execbench/core/data/workload.py`, `Solution` in `src/sol_execbench/core/data/solution.py`, `Trace` in `src/sol_execbench/core/data/trace.py`.
- Pattern: Pydantic models with validators; JSON and JSONL files are parsed into models before being used.

**Build And Staging:**
- Purpose: Convert validated models and source strings into an executable staging directory.
- Examples: `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py`, `build_ext.py` in `src/sol_execbench/driver/templates/build_ext.py`.
- Pattern: Write normalized files first, then return commands for the CLI to run with explicit `cwd`.

**ROCm Language Boundary:**
- Purpose: Enforce ROCm-only solution categories and native/Python separation.
- Examples: `SupportedLanguages`, `SupportedHardware`, and `BuildSpec` validators in `src/sol_execbench/core/data/solution.py`.
- Pattern: Reject legacy CUDA/NVIDIA schema values during validation; use HIP names such as `hip_cflags`.

**Trace Authority:**
- Purpose: Record per-workload correctness, performance, environment, and failure status.
- Examples: `EvaluationStatus`, `Evaluation`, and `Trace` in `src/sol_execbench/core/data/trace.py`.
- Pattern: The generated driver emits strict JSON with `allow_nan=False`; optional sidecars are diagnostic and do not mutate the trace schema.

**Evidence Reports:**
- Purpose: Make compatibility, runtime, consistency, claim, stability, trust, and scoring evidence explicit.
- Examples: `RocmCompatibilityMatrixReport` in `src/sol_execbench/core/compatibility.py`, `EnvironmentDiagnostics` in `src/sol_execbench/core/environment.py`, `ToolchainRoutingReport` in `src/sol_execbench/core/toolchain.py`, `TrustSummaryReport` in `src/sol_execbench/core/trust_summary.py`.
- Pattern: Pydantic report model plus `build_*`, `render_*_markdown`, and `write_*_reports` helpers.

**AMD Bound And Score Graphs:**
- Purpose: Estimate AMD SOL bounds and score traces against derived hardware-aware constraints.
- Examples: `BoundGraph` in `src/sol_execbench/core/scoring/amd_bound_graph.py`, `OperatorWorkEstimate` in `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `AmdSolBoundV2Artifact` in `src/sol_execbench/core/scoring/amd_sol_v2.py`, `AmdNativeScore` in `src/sol_execbench/core/scoring/amd_score.py`.
- Pattern: Build a graph from definition/workload, estimate work per op family, aggregate into artifacts, then attach evidence refs.

## Entry Points

**Main Evaluator CLI:**
- Location: `src/sol_execbench/cli/main.py`
- Triggers: `sol-execbench` console script from `pyproject.toml`.
- Responsibilities: Evaluation, contract, doctor, toolchain routing, sidecar writing.

**Baseline CLI:**
- Location: `src/sol_execbench/cli/baseline.py`
- Triggers: `sol-execbench-baseline` console script from `pyproject.toml`.
- Responsibilities: Compare trace JSONL against baseline trace JSONL using `src/sol_execbench/core/baseline.py`.

**Dataset Runner:**
- Location: `scripts/run_dataset.py`
- Triggers: `uv run scripts/run_dataset.py ...`.
- Responsibilities: Batch problem discovery, solution wrapping, CLI invocation, trace inspection, timing evidence, derived score/evidence reports.

**Report Scripts:**
- Location: `scripts/report_*.py`, `scripts/diff_matrix_reports.py`, `scripts/export_matrix_schema.py`, `scripts/inspect_dataset.py`
- Triggers: Direct `uv run scripts/<name>.py ...`.
- Responsibilities: Thin command wrappers over `src/sol_execbench/core/dataset/`, `src/sol_execbench/core/scoring/`, and evidence/reporting modules.

**Docker Environment:**
- Location: `scripts/run_docker.sh`, `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`
- Triggers: `./scripts/run_docker.sh --build`.
- Responsibilities: ROCm container build/run, target selection, device-node setup, clock-lock handling.

**Generated Runtime Driver:**
- Location: `src/sol_execbench/driver/templates/eval_driver.py`
- Triggers: Copied by `ProblemPackager.execute()` and run as `python eval_driver.py` in staging.
- Responsibilities: Actual workload execution and JSON trace emission.

## Architectural Constraints

- **Threading:** The evaluator CLI is synchronous and uses subprocesses. The generated driver checks active thread counts around user execution to detect reward hacks (`src/sol_execbench/driver/templates/eval_driver.py:427`).
- **Global state:** The generated driver intentionally mutates process stdout before importing Torch so non-JSON library output goes to stderr (`src/sol_execbench/driver/templates/eval_driver.py:36`). It also inserts the staging directory at the front of `sys.path` (`src/sol_execbench/driver/templates/eval_driver.py:45`).
- **Circular imports:** No known circular dependency chain is required by the architecture. Keep `src/sol_execbench/core/data/` independent from CLI and driver modules.
- **Subprocess isolation:** User solution code must run in `eval_driver.py`, not inside `src/sol_execbench/cli/main.py`.
- **ROCm-only schema:** New solution categories must be added through `SupportedLanguages` and validators in `src/sol_execbench/core/data/solution.py`; do not reintroduce CUDA category names.
- **JSONL trace contract:** CLI and dataset tooling assume one valid JSON trace object per stdout line when `--json` is used.
- **Sidecar authority:** Environment snapshots, profiler metadata, and static evidence are diagnostic sidecars. They must not become required for `Trace` schema validity.
- **Generated files:** The driver templates in `src/sol_execbench/driver/templates/` are source templates copied into staging; changes there affect runtime behavior, not only packaging behavior.

## Anti-Patterns

### Running User Code In The CLI Process

**What happens:** New evaluator logic imports or executes a submitted solution from `src/sol_execbench/cli/main.py`.
**Why it's wrong:** It bypasses staging cleanup, stdout isolation, reward-hack checks, and subprocess failure containment.
**Do this instead:** Add runtime behavior to `src/sol_execbench/driver/templates/eval_driver.py` or helper modules in `src/sol_execbench/core/bench/`, then invoke through `ProblemPackager.execute()`.

### Duplicating JSON Parsing In Scripts

**What happens:** A script directly interprets benchmark schemas using ad hoc dictionaries after the first file read.
**Why it's wrong:** It drifts from validators in `Definition`, `Workload`, `Solution`, and report models.
**Do this instead:** Parse into models from `src/sol_execbench/core/data/` or call existing dataset/scoring helpers from `src/sol_execbench/core/dataset/` and `src/sol_execbench/core/scoring/`.

### Treating Diagnostic Evidence As Benchmark Authority

**What happens:** Static evidence, profiler output, or environment probes are used to override per-workload correctness status.
**Why it's wrong:** The trace contract keeps correctness and timing authority in the generated driver; sidecars are optional and may be unavailable.
**Do this instead:** Add sidecar report models and warnings in `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, or report modules, while leaving `Trace` status semantics in `src/sol_execbench/core/data/trace.py`.

### Adding CUDA Compatibility Paths

**What happens:** New code accepts `cuda_cpp`, `cutlass`, `cudnn`, `cuda_cflags`, or NVIDIA-only score assumptions.
**Why it's wrong:** The repository is a ROCm-only port and validators explicitly reject legacy CUDA/NVIDIA schema values.
**Do this instead:** Use `hip_cpp`, `hipblas`, `miopen`, `ck`, `rocwmma`, and `hip_cflags` in `src/sol_execbench/core/data/solution.py`, with migration docs under `docs/`.

## Error Handling

**Strategy:** Validate schemas early, isolate runtime failures into subprocess traces, and make diagnostic evidence nonfatal unless the requested benchmark semantics require failure.

**Patterns:**
- Raise `click.ClickException` for CLI argument and file-resolution errors in `src/sol_execbench/cli/main.py`.
- Convert staged runtime failures into `Trace` rows with `EvaluationStatus` values in `src/sol_execbench/driver/templates/eval_driver.py`.
- Exit the CLI with code `1` when compilation fails, no traces are produced, or any trace is not `PASSED` (`src/sol_execbench/cli/main.py:769`).
- Catch optional environment/profile/static-evidence sidecar failures and report warnings without changing correctness (`src/sol_execbench/cli/main.py`).
- Use explicit report status/reason models in dataset, compatibility, and scoring modules.

## Cross-Cutting Concerns

**Logging:** CLI user output uses Rich `Console(stderr=True)` in `src/sol_execbench/cli/main.py`. Generated driver redirects non-JSON stdout to stderr and writes only strict trace JSON to the saved stdout file descriptor. Dataset scripts print progress and write bounded CLI failure logs under their output directories.

**Validation:** Pydantic validators enforce schema and ROCm constraints in `src/sol_execbench/core/data/`. Report modules define their own Pydantic models for compatibility, runtime, trust, consistency, stability, and scoring artifacts.

**Authentication:** Not applicable. The package is a local benchmark CLI. Hugging Face dataset access or external downloads are handled by scripts and user environment, not by an in-app identity layer.

---

*Architecture analysis: 2026-06-01*
