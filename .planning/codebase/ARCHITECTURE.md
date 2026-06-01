<!-- refreshed: 2026-06-01 -->
# Architecture

**Analysis Date:** 2026-06-01

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    User And Batch Entry Points               │
├──────────────────┬──────────────────┬───────────────────────┤
│   Evaluator CLI  │  Baseline CLI    │   Dataset Scripts     │
│ `src/sol_execbench/cli/main.py` │ `src/sol_execbench/cli/baseline.py` │ `scripts/run_dataset.py` │
└────────┬─────────┴────────┬─────────┴──────────┬────────────┘
         │                  │                     │
         ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Typed Core Contracts And Services               │
│ `src/sol_execbench/core/data/`, `core/dataset/`, `core/scoring/` │
└────────┬─────────────────────────────┬──────────────────────┘
         │                             │
         ▼                             ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│     Staging And Execution   │ │   Evidence And Reports      │
│ `src/sol_execbench/driver/` │ │ `src/sol_execbench/core/`   │
└────────┬────────────────────┘ └──────────────┬──────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Temp Staging Dir, Trace JSONL, Sidecars, Dataset Outputs    │
│  `tempfile.mkdtemp(prefix="sol_execbench_")`, `out/`, `data/` │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Evaluator CLI | Resolve problem inputs, load models, run compile/evaluation subprocesses, parse traces, write sidecars, and dispatch metadata subcommands. | `src/sol_execbench/cli/main.py` |
| Baseline CLI | Compare candidate trace JSONL against baseline trace JSONL and emit text or JSON reports. | `src/sol_execbench/cli/baseline.py` |
| Problem packager | Normalize problem files and solution sources into a staging directory and return compile/evaluate commands. | `src/sol_execbench/driver/problem_packager.py` |
| Build template | Build native HIP/C++ extension sources into `benchmark_kernel.so` using `torch.utils.cpp_extension`. | `src/sol_execbench/driver/templates/build_ext.py` |
| Evaluation template | Run reference and user code in the staged subprocess and emit one `Trace` JSON object per workload. | `src/sol_execbench/driver/templates/eval_driver.py` |
| Data contracts | Define Pydantic schemas for definitions, workloads, solutions, traces, dtypes, shape expressions, and evaluator contract metadata. | `src/sol_execbench/core/data/` |
| Benchmark helpers | Generate inputs, allocate outputs, validate correctness, time kernels, collect profiler/static evidence, and enforce reward-hack guardrails. | `src/sol_execbench/core/bench/` |
| Dataset services | Inspect dataset layout, build manifests, compute readiness, select ready subsets, track execution closure, and support deterministic sharding. | `src/sol_execbench/core/dataset/` |
| Scoring services | Compute SOL score, AMD-native score reports, AMD bound estimates, SOLAR derivation evidence, and baseline artifacts. | `src/sol_execbench/core/scoring/`, `src/sol_execbench/sol_score.py` |
| Evidence services | Collect ROCm environment snapshots, toolchain routing reports, compatibility matrices, runtime evidence, consistency, stability, and trust summaries. | `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/compatibility.py` |
| Dataset runner script | Provide the high-level command-line loop for single-problem and dataset-batch runs. | `scripts/run_dataset.py` |

## Pattern Overview

**Overall:** Layered CLI application with typed contracts, subprocess staging, and sidecar evidence.

**Key Characteristics:**
- Keep user-supplied solution execution outside the CLI process by staging files with `ProblemPackager` and running generated scripts from `src/sol_execbench/driver/templates/`.
- Use Pydantic models in `src/sol_execbench/core/data/` as the public schema boundary for definition, workload, solution, and trace data.
- Keep benchmark correctness and trace JSONL canonical, while optional environment, profiler, static-kernel, scoring, and closure evidence live in separate sidecars.
- Keep `scripts/` thin where possible by delegating reusable dataset and evidence logic to `src/sol_execbench/core/dataset/` and `src/sol_execbench/core/scoring/`.

## Layers

**CLI Layer:**
- Purpose: Expose console commands, parse user options, print human-facing summaries, and coordinate subprocesses.
- Location: `src/sol_execbench/cli/`
- Contains: Click commands, Rich table output, metadata subcommand dispatch, baseline comparison CLI.
- Depends on: `src/sol_execbench/core/`, `src/sol_execbench/driver/`
- Used by: `pyproject.toml` console scripts `sol-execbench` and `sol-execbench-baseline`.

**Typed Contract Layer:**
- Purpose: Validate and serialize benchmark schemas and evidence payloads.
- Location: `src/sol_execbench/core/data/`, plus evidence models in `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, and `src/sol_execbench/core/compatibility.py`.
- Contains: Pydantic models, enums, validators, shape/dtype helpers, JSON utility functions.
- Depends on: Pydantic, Python stdlib; `src/sol_execbench/core/data/dtypes.py` imports torch lazily for dtype conversion.
- Used by: CLI, driver templates, dataset services, scoring, reports, tests, examples.

**Driver And Staging Layer:**
- Purpose: Materialize an isolated working directory for each benchmark run and generate commands for native compile and evaluation.
- Location: `src/sol_execbench/driver/`
- Contains: `ProblemPackager`, generated `build_ext.py`, generated `eval_driver.py`.
- Depends on: Core data models, subprocess, ROCm discovery helpers.
- Used by: `src/sol_execbench/cli/main.py` and driver-focused tests under `tests/sol_execbench/driver/`.

**Benchmark Runtime Layer:**
- Purpose: Execute workload correctness and timing inside the generated evaluation subprocess.
- Location: `src/sol_execbench/core/bench/`
- Contains: Input generation, output normalization, correctness metrics, event timing, clock-lock checks, reward-hack checks, profiling, static-kernel evidence.
- Depends on: PyTorch ROCm, safetensors, core data models, ROCm command-line tools when optional evidence is requested.
- Used by: `src/sol_execbench/driver/templates/eval_driver.py`, `scripts/run_dataset.py`, tests under `tests/sol_execbench/core/bench/`.

**Dataset Layer:**
- Purpose: Manage downloaded SOL ExecBench dataset structure and batch-run evidence lifecycle.
- Location: `src/sol_execbench/core/dataset/`
- Contains: Category validation, layout inspection, checksums, inventory, manifest, readiness, ready subset, run state, execution closure, evidence references, sharding, parity-gap and paper-denominator reports.
- Depends on: Core data models, filesystem paths, stable JSON checksums.
- Used by: `scripts/run_dataset.py`, `scripts/inspect_dataset.py`, `scripts/download_solexecbench.py`, report scripts, tests under `tests/sol_execbench/`.

**Scoring And Reporting Layer:**
- Purpose: Convert canonical traces and derived evidence into guarded score and trust reports.
- Location: `src/sol_execbench/core/scoring/`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/baseline.py`, `src/sol_execbench/sol_score.py`
- Contains: SOL formula, AMD hardware models, AMD bound estimates, SOLAR derivation, AMD-native score reports, baseline artifacts, trace run summaries.
- Depends on: Core data models, packaged hardware model data in `src/sol_execbench/data/amd_hardware_models/`.
- Used by: Dataset runner, report scripts, baseline CLI, scoring tests.

**Operational Assets Layer:**
- Purpose: Provide executable examples, documentation, Docker environment, and downloaded benchmark assets.
- Location: `examples/`, `docs/`, `docker/`, `data/`
- Contains: Example problem directories, user documentation, Dockerfile/entrypoint, local downloaded benchmark data.
- Depends on: Package code and ROCm runtime.
- Used by: Users, tests under `tests/examples/` and `tests/docker/dependencies/`.

## Data Flow

### Primary Request Path

1. Console script `sol-execbench` dispatches to `sol_execbench.cli:cli` (`pyproject.toml`).
2. `SolExecbenchCli.main()` routes `contract`, `doctor`, and `toolchain` subcommands before falling back to `_evaluate_cli()` (`src/sol_execbench/cli/main.py`).
3. `_evaluate_cli()` resolves either `<problem_dir>` or explicit `--definition`, `--workload`, `--solution`, and optional `--config` paths (`src/sol_execbench/cli/main.py`).
4. `_load_definition()`, `_load_workloads()`, `_load_solution()`, and `_load_config()` create `Definition`, `Workload`, `Solution`, and `BenchmarkConfig` objects (`src/sol_execbench/cli/main.py`).
5. `ProblemPackager.__init__()` writes `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, and solution sources into `tempfile.mkdtemp(prefix="sol_execbench_")` (`src/sol_execbench/driver/problem_packager.py`).
6. Native ROCm solutions run `ProblemPackager.compile()` and the generated `build_ext.py` to produce `benchmark_kernel.so` (`src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`).
7. `ProblemPackager.execute()` stages `eval_driver.py`; `_run_evaluation_command()` runs it with `PYTORCH_ALLOC_CONF=expandable_segments:True` (`src/sol_execbench/cli/main.py`).
8. The evaluation driver loads staged problem data, imports reference code, imports user Python or native module code, validates correctness, measures latency, and emits strict JSONL through `emit_trace_jsonl()` (`src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/eval_runtime.py`).
9. `ProblemPackager.convert_stdout_to_traces()` parses trace JSONL into `Trace` models; `_evaluate_cli()` writes optional trace output and sidecars, prints or emits JSON, closes staging, and exits based on all trace statuses (`src/sol_execbench/cli/main.py`).

### Dataset Batch Flow

1. `scripts/run_dataset.py` receives either a single problem directory or a dataset root with category directories.
2. `discover_problems()` delegates to `src/sol_execbench/core/dataset/run_state.py` to find directories containing `definition.json` and `workload.jsonl`.
3. `build_solution_for_problem()` chooses a problem-provided solution JSON, wraps a custom `solution.py`, or wraps `Definition.reference` as a PyTorch solution (`src/sol_execbench/core/dataset/runner.py`).
4. `build_cli_command()` builds a `sol-execbench --definition ... --workload ... --solution ... --json` subprocess command (`src/sol_execbench/core/dataset/runner.py`).
5. `run_cli()` captures stdout/stderr, parses trace dictionaries, and writes bounded CLI logs on failures (`src/sol_execbench/core/dataset/runner.py`).
6. Dataset closure and reuse records are built through `src/sol_execbench/core/dataset/run_closure.py` and `src/sol_execbench/core/dataset/run_state.py`.
7. Optional AMD score, AMD SOL bound, SOLAR derivation, and timing-evidence sidecars are generated through `src/sol_execbench/core/dataset/runner.py` and `src/sol_execbench/core/scoring/`.

### Metadata And Evidence Flow

1. `sol-execbench contract --json` emits `build_evaluator_contract()` output from `src/sol_execbench/core/data/contract.py`.
2. `sol-execbench doctor --json` emits `build_environment_diagnostics()` output from `src/sol_execbench/core/environment.py`.
3. `sol-execbench toolchain --json` emits `build_toolchain_routing_report()` output from `src/sol_execbench/core/toolchain.py`.
4. `--profile rocprofv3` runs the staged evaluator under `collect_rocprofv3_profile()` and writes metadata through `_write_profile_sidecar()` (`src/sol_execbench/cli/main.py`, `src/sol_execbench/core/bench/rocm_profiler.py`).
5. `--static-evidence auto` collects native build artifacts and runs static extractors through `src/sol_execbench/core/bench/static_kernel_evidence.py`.
6. `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` control environment snapshot sidecars in `src/sol_execbench/cli/main.py`.

**State Management:**
- Benchmark request state is explicit in Pydantic objects and staged files.
- Evaluation subprocess state lives in a temporary staging directory created by `ProblemPackager`.
- Optional dataset-run state is persisted as JSON traces, summaries, execution-closure reports, sidecars, and CLI logs under output directories.
- Module-level constants define schema versions, environment variable names, tool IDs, and supported category sets.

## Key Abstractions

**Problem Schemas:**
- Purpose: Public benchmark contract for problem definitions, workloads, solutions, and output traces.
- Examples: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/trace.py`
- Pattern: Pydantic models with validators that enforce schema compatibility and ROCm-specific guardrails.

**ProblemPackager:**
- Purpose: Convert typed models into a runnable staging directory.
- Examples: `src/sol_execbench/driver/problem_packager.py`
- Pattern: Context-manageable class that owns lifecycle cleanup and emits command lists rather than executing internally.

**Generated Driver Templates:**
- Purpose: Keep subprocess evaluation code versioned while writing a self-contained script into each staging directory.
- Examples: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`
- Pattern: Source templates copied into staging and executed with `python`.

**Benchmark Helpers:**
- Purpose: Provide importable runtime operations used by the generated driver.
- Examples: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/eval_runtime.py`
- Pattern: Pure helpers plus torch-backed functions with injectable timing/probe seams for tests.

**Dataset Reports:**
- Purpose: Model dataset layout, readiness, closure, parity, denominator, and sharding artifacts.
- Examples: `src/sol_execbench/core/dataset/layout.py`, `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/dataset/sharding.py`
- Pattern: Filesystem scanners and small artifact builders returning Pydantic models or plain JSON-compatible dicts.

**Derived Scoring:**
- Purpose: Produce guarded AMD-native score artifacts without changing canonical trace semantics.
- Examples: `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/sol_score.py`
- Pattern: Dataclasses/Pydantic-like artifacts with explicit warnings and evidence references.

## Entry Points

**Evaluator CLI:**
- Location: `src/sol_execbench/cli/main.py`
- Triggers: `sol-execbench` console script from `pyproject.toml`
- Responsibilities: Evaluate one problem, emit contract/doctor/toolchain metadata, manage optional profiler/static/environment sidecars.

**Baseline CLI:**
- Location: `src/sol_execbench/cli/baseline.py`
- Triggers: `sol-execbench-baseline` console script from `pyproject.toml`
- Responsibilities: Compare candidate traces to one or more baselines.

**Dataset Runner:**
- Location: `scripts/run_dataset.py`
- Triggers: `uv run scripts/run_dataset.py <problem-or-dataset-dir>`
- Responsibilities: Discover problems, build reference/custom solutions, invoke `sol-execbench`, aggregate summaries, closure, and optional evidence reports.

**Dataset Inspect/Report Scripts:**
- Location: `scripts/inspect_dataset.py`, `scripts/report_paper_denominator.py`, `scripts/report_parity_gaps.py`, `scripts/export_matrix_schema.py`, `scripts/diff_matrix_reports.py`
- Triggers: Direct `uv run scripts/<script>.py ...`
- Responsibilities: Build dataset, matrix, and evidence reports using `src/sol_execbench/core/`.

**Docker Runtime:**
- Location: `scripts/run_docker.sh`, `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`
- Triggers: `./scripts/run_docker.sh --build`
- Responsibilities: Build and enter ROCm-capable container workflows.

## Architectural Constraints

- **Threading:** Evaluation is mostly single-process orchestration plus subprocesses. The generated evaluator checks thread counts around timing through `check_thread_injection()` in `src/sol_execbench/core/bench/reward_hack.py`; dataset tests use pytest-xdist by default from `pyproject.toml`.
- **Global state:** The generated evaluation driver redirects stdout before importing torch, stores `_real_stdout`, inserts the staging directory into `sys.path`, registers synthetic package chains for user modules, and holds runtime globals in `src/sol_execbench/driver/templates/eval_driver.py`.
- **Circular imports:** Core packages expose convenience barrels in `src/sol_execbench/core/__init__.py` and `src/sol_execbench/core/data/__init__.py`; new deep modules should import concrete modules when that avoids broad import chains.
- **Subprocess boundary:** The CLI subprocess boundary is an execution guardrail, not a hardened sandbox. Untrusted code still requires external isolation such as Docker, VM, or dedicated ROCm host.
- **Canonical traces:** `Trace` JSONL from `src/sol_execbench/core/data/trace.py` is the benchmark output authority. Optional sidecars are diagnostic or derived evidence and must not be treated as trace schema changes.
- **ROCm-only schema:** `BuildSpec` rejects legacy CUDA/NVIDIA language and compile-option values in `src/sol_execbench/core/data/solution.py`.
- **Native compile boundary:** HIP/C++ compilation is staged and driven through `torch.utils.cpp_extension` in `src/sol_execbench/driver/templates/build_ext.py`.

## Anti-Patterns

### Running User Code In CLI Process

**What happens:** Importing candidate solution modules directly from `src/sol_execbench/cli/main.py` would execute user code in the orchestrator process.
**Why it's wrong:** The current architecture keeps user/reference code in a staged subprocess so stdout framing, imports, native artifacts, and cleanup do not contaminate CLI state.
**Do this instead:** Stage through `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py` and add runtime behavior to `src/sol_execbench/driver/templates/eval_driver.py` or helpers in `src/sol_execbench/core/bench/`.

### Changing Trace Schema For Optional Evidence

**What happens:** Adding profiler, static-kernel, environment, or AMD-score fields directly to `Trace` for diagnostic needs.
**Why it's wrong:** `Trace` in `src/sol_execbench/core/data/trace.py` is the canonical benchmark output; optional evidence has separate claim boundaries.
**Do this instead:** Add sidecar models or writers near `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/environment.py`, or `src/sol_execbench/core/scoring/`.

### Expanding `scripts/run_dataset.py` With Reusable Logic

**What happens:** Placing reusable dataset state, closure, scoring, or evidence algorithms directly into the script.
**Why it's wrong:** The repo already centralizes importable dataset logic under `src/sol_execbench/core/dataset/` and `src/sol_execbench/core/scoring/`; scripts are command-line glue.
**Do this instead:** Add reusable behavior to `src/sol_execbench/core/dataset/runner.py`, `src/sol_execbench/core/dataset/run_state.py`, `src/sol_execbench/core/dataset/run_closure.py`, or `src/sol_execbench/core/scoring/`, then call it from `scripts/run_dataset.py`.

### Importing Through Broad Barrels In Low-Level Modules

**What happens:** Deep helpers import all public models through `sol_execbench.core` when they only need one concrete model.
**Why it's wrong:** Barrel imports in `src/sol_execbench/core/__init__.py` are convenient for public API and CLI code but can widen import dependencies in low-level modules.
**Do this instead:** Use concrete imports such as `from sol_execbench.core.data.trace import Trace` in new low-level modules, following `src/sol_execbench/core/dataset/runner.py`.

## Error Handling

**Strategy:** Validate schemas eagerly with Pydantic, keep subprocess failures bounded and diagnostic, emit typed per-workload statuses where possible, and keep noncanonical evidence failures nonfatal.

**Patterns:**
- CLI input errors use `click.ClickException` in `src/sol_execbench/cli/main.py`.
- Definition, workload, solution, and trace validation errors come from Pydantic models under `src/sol_execbench/core/data/`.
- Evaluation failures inside the generated driver emit `Trace` records with `EvaluationStatus` values such as `RUNTIME_ERROR`, `INVALID_REFERENCE`, `INCORRECT_NUMERICAL`, and `REWARD_HACK` from `src/sol_execbench/core/data/trace.py`.
- CLI-level no-trace failures write bounded diagnostic sidecars with `_write_no_trace_diagnostics_sidecar()` in `src/sol_execbench/cli/main.py`.
- Optional profiler/static/environment collection catches exceptions and writes warning-bearing sidecars instead of mutating benchmark correctness status.
- Dataset subprocess failures write bounded CLI logs through `save_cli_log()` and `save_cli_timeout_log()` in `src/sol_execbench/core/dataset/runner.py`.

## Cross-Cutting Concerns

**Logging:** Human-facing CLI output uses Rich in `src/sol_execbench/cli/main.py`; dataset scripts use `print()` in `scripts/run_dataset.py` and `src/sol_execbench/core/dataset/runner.py`; bounded subprocess logs are persisted as sidecar/log files.

**Validation:** Pydantic validators enforce public schemas in `src/sol_execbench/core/data/`; dataset and evidence artifacts use Pydantic models or JSON-compatible dataclasses across `src/sol_execbench/core/dataset/`, `src/sol_execbench/core/environment.py`, and `src/sol_execbench/core/toolchain.py`.

**Authentication:** Not applicable for benchmark execution. Dataset download helpers may access external services through user-local tooling, but no auth provider is implemented in package code.

**Security:** Source paths, compile options, language categories, CUDA legacy values, reward-hack patterns, lazy outputs, monkey patching, thread injection, and dynamic C++ extension loads are guarded in `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/bench/reward_hack.py`, and `src/sol_execbench/core/bench/eval_runtime.py`.

---

*Architecture analysis: 2026-06-01*
