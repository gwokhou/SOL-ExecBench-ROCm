<!-- refreshed: 2026-05-28 -->
# Architecture

**Analysis Date:** 2026-05-28

## System Overview

```text
+-------------------------------------------------------------+
|                         CLI Layer                           |
| `src/sol_execbench/cli/main.py`                             |
| `src/sol_execbench/cli/baseline.py`                         |
+---------------------+-------------------+-------------------+
                      |                   |
                      v                   v
+-------------------------------------------------------------+
|                       Core Domain                           |
| Schemas: `src/sol_execbench/core/data/`                     |
| Bench:   `src/sol_execbench/core/bench/`                    |
| Score:   `src/sol_execbench/core/scoring/`                  |
| Dataset: `src/sol_execbench/core/dataset/`                  |
+---------------------+-------------------+-------------------+
                      |                   |
                      v                   v
+-------------------------------------------------------------+
|                 Execution Isolation Layer                   |
| `src/sol_execbench/driver/problem_packager.py`              |
| `src/sol_execbench/driver/templates/build_ext.py`           |
| `src/sol_execbench/driver/templates/eval_driver.py`         |
+---------------------+-------------------+-------------------+
                      |                   |
                      v                   v
+-------------------------------------------------------------+
|                Outputs, Evidence, and Assets                |
| Trace JSONL, optional sidecars, `data/`, `examples/`        |
| Docker targets in `docker/rocm-targets.json`                |
+-------------------------------------------------------------+
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Main CLI | Resolve problem inputs, load schemas, run compile/evaluate subprocesses, emit trace JSONL or Rich tables, and dispatch metadata subcommands. | `src/sol_execbench/cli/main.py` |
| Baseline CLI | Compare candidate trace JSONL files against baseline trace JSONL files. | `src/sol_execbench/cli/baseline.py` |
| Public core exports | Re-export stable model, environment, toolchain, and bench-config APIs for internal callers and generated drivers. | `src/sol_execbench/core/__init__.py` |
| Schema models | Define validated public benchmark contracts for definitions, solutions, workloads, traces, and evaluator metadata. | `src/sol_execbench/core/data/` |
| Benchmark primitives | Generate inputs, check correctness, measure GPU latency, enforce clock policies, and detect reward-hacking behavior. | `src/sol_execbench/core/bench/` |
| Problem packager | Materialize a benchmark problem into an isolated staging directory and return subprocess commands for compilation and evaluation. | `src/sol_execbench/driver/problem_packager.py` |
| Build template | Compile native HIP/C++ solution sources into `benchmark_kernel.so` through `torch.utils.cpp_extension`. | `src/sol_execbench/driver/templates/build_ext.py` |
| Evaluation template | Import reference and user code in the staging process, run correctness and timing loops, and emit strict JSON trace lines. | `src/sol_execbench/driver/templates/eval_driver.py` |
| Scoring | Build derived AMD-native score reports and guarded SOL-relative interpretations from trace and bound artifacts. | `src/sol_execbench/core/scoring/` |
| Dataset utilities | Inspect, acquire, checksum, and classify SOL ExecBench dataset layouts without running GPU evaluation. | `src/sol_execbench/core/dataset/` |
| Environment evidence | Collect optional ROCm, PyTorch, and toolchain diagnostics without changing canonical trace schema. | `src/sol_execbench/core/environment.py` |
| Docker/runtime matrix | Model ROCm Docker targets, dependency observations, and runtime evidence for validation workflows. | `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/runtime_evidence.py` |
| Dataset batch runner | Orchestrate many problem executions and generate derived sidecars from CLI traces. | `scripts/run_dataset.py` |

## Pattern Overview

**Overall:** CLI-driven benchmark harness with schema-first domain models, generated staging scripts, and sidecar evidence.

**Key Characteristics:**
- Treat `Definition`, `Solution`, `Workload`, and `Trace` Pydantic models in `src/sol_execbench/core/data/` as the canonical public contract.
- Keep untrusted or user-provided solution execution in a temporary staging directory created by `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py`.
- Keep optional evidence, diagnostics, profiling, static kernel data, and derived scores outside canonical trace JSONL semantics.
- Use PyTorch ROCm compatibility APIs such as `torch.cuda.Event` and `torch.cuda.is_available()` while documenting ROCm intent in wrappers.
- Prefer deterministic JSON sidecars and stable schema version constants for artifacts in `src/sol_execbench/core/*`.

## Layers

**CLI Layer:**
- Purpose: Parse user commands, load files, start subprocesses, write outputs, and expose metadata commands.
- Location: `src/sol_execbench/cli/`
- Contains: Click commands, Rich table rendering, staging lifecycle orchestration, sidecar output helpers.
- Depends on: `src/sol_execbench/core/`, `src/sol_execbench/driver/`, Python `subprocess`, ROCm profiler helpers.
- Used by: Console scripts declared in `pyproject.toml`, tests in `tests/sol_execbench/`, scripts in `scripts/`.

**Domain Schema Layer:**
- Purpose: Validate and serialize the public benchmark objects.
- Location: `src/sol_execbench/core/data/`
- Contains: Pydantic models for `Definition`, `Solution`, `Workload`, `Trace`, evaluator contract, dtypes, shapes, JSON helpers.
- Depends on: Pydantic, Python AST for reference validation, local dtype and shape helpers.
- Used by: CLI, generated driver templates, packager, scoring, dataset tests, examples.

**Benchmark Runtime Layer:**
- Purpose: Execute correctness and performance measurement logic shared by generated drivers.
- Location: `src/sol_execbench/core/bench/`
- Contains: Input generation, safetensors loading, output normalization, tolerance checking, timing, clock lock checks, reward-hack detection, static kernel evidence, rocprofv3 evidence.
- Depends on: PyTorch ROCm APIs, model classes in `src/sol_execbench/core/data/`, optional ROCm tools.
- Used by: `src/sol_execbench/driver/templates/eval_driver.py` and CLI evidence collection.

**Execution Isolation Layer:**
- Purpose: Convert in-memory benchmark objects into a staging directory that can be compiled and evaluated through subprocesses.
- Location: `src/sol_execbench/driver/`
- Contains: `ProblemPackager`, `build_ext.py`, `eval_driver.py`.
- Depends on: Core schemas, PyTorch extension API, generated JSON files and user source files.
- Used by: `src/sol_execbench/cli/main.py` and driver tests.

**Evidence And Reporting Layer:**
- Purpose: Produce optional environment snapshots, profiler sidecars, static kernel sidecars, derived reports, and guarded score interpretations.
- Location: `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/scoring/`
- Contains: Versioned artifact models, probe runners, report builders, score guards.
- Depends on: Canonical traces, bounded subprocess probes, deterministic serialization.
- Used by: CLI options, dataset runner, docs/tests asserting claim boundaries.

**Dataset And Tooling Layer:**
- Purpose: Manage dataset acquisition/readiness and ROCm target/dependency matrix reporting.
- Location: `src/sol_execbench/core/dataset/`, `scripts/`, `docker/`
- Contains: Dataset layout inspectors, manifests, readiness classifiers, Docker target manifests, batch runner.
- Depends on: Local filesystem layout, Hugging Face dataset rows in download scripts, core schemas.
- Used by: `scripts/download_solexecbench.py`, `scripts/inspect_dataset.py`, `scripts/run_dataset.py`, Docker tests.

## Data Flow

### Primary Request Path

1. User invokes `sol-execbench` through the console script declared in `pyproject.toml:28`; `src/sol_execbench/cli/__init__.py` exposes `cli`.
2. `SolExecbenchCli.main()` dispatches metadata subcommands or forwards to `_evaluate_cli()` (`src/sol_execbench/cli/main.py:851`, `src/sol_execbench/cli/main.py:891`).
3. `_evaluate_cli()` resolves either a problem directory or explicit file paths, then loads `Definition`, `Workload`, `Solution`, and `BenchmarkConfig` (`src/sol_execbench/cli/main.py:596`, `src/sol_execbench/cli/main.py:613`).
4. `ProblemPackager` writes `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, and solution sources to a temporary staging directory (`src/sol_execbench/driver/problem_packager.py:91`, `src/sol_execbench/driver/problem_packager.py:112`).
5. For native ROCm languages, `compile()` injects target architecture flags, writes `build_ext.py`, and returns `["python", "build_ext.py"]`; the CLI runs it in the staging directory (`src/sol_execbench/driver/problem_packager.py:175`, `src/sol_execbench/cli/main.py:644`).
6. `execute()` writes `eval_driver.py` and returns `["python", "eval_driver.py"]`; the CLI runs it directly or under optional `rocprofv3` collection (`src/sol_execbench/driver/problem_packager.py:203`, `src/sol_execbench/cli/main.py:692`).
7. `eval_driver.py` redirects non-JSON output to stderr, loads staged problem objects, imports reference and user code, applies reward-hack checks, runs 10 correctness rounds, measures latency with `time_runnable()`, and emits one strict JSON `Trace` per workload (`src/sol_execbench/driver/templates/eval_driver.py:38`, `src/sol_execbench/driver/templates/eval_driver.py:333`, `src/sol_execbench/driver/templates/eval_driver.py:677`).
8. The CLI parses stdout JSONL into `Trace` models and writes trace output plus optional environment, profile, and static evidence sidecars (`src/sol_execbench/driver/problem_packager.py:228`, `src/sol_execbench/cli/main.py:737`, `src/sol_execbench/cli/main.py:746`).

### Metadata And Diagnostics Flow

1. `sol-execbench contract --json` routes through `SolExecbenchCli.main()` to `_contract_cli()` (`src/sol_execbench/cli/main.py:861`, `src/sol_execbench/cli/main.py:771`).
2. `_contract_cli()` serializes `build_evaluator_contract()` without requiring a problem directory (`src/sol_execbench/cli/main.py:778`, `src/sol_execbench/core/data/contract.py`).
3. `sol-execbench doctor --json` calls `build_environment_diagnostics()`, which combines bounded ROCm tool probes and PyTorch ROCm smoke checks (`src/sol_execbench/cli/main.py:782`, `src/sol_execbench/core/environment.py:249`).
4. `sol-execbench toolchain --json` builds routing reports from `ToolchainRoutingRequest` and `default_toolchain_registry()` (`src/sol_execbench/cli/main.py:793`, `src/sol_execbench/cli/main.py:837`).

### Dataset Batch And Scoring Flow

1. `scripts/run_dataset.py` discovers benchmark problem directories, builds CLI commands, and invokes `sol-execbench` as a subprocess (`scripts/run_dataset.py:95`, `scripts/run_dataset.py:246`, `scripts/run_dataset.py:279`).
2. The script reads trace payloads into `Trace` models and may collect timing evidence, execution closure records, AMD SOL artifacts, and derived score reports (`scripts/run_dataset.py:333`, `scripts/run_dataset.py:502`, `scripts/run_dataset.py:917`).
3. Scoring functions use canonical trace performance data and bound/baseline artifacts to build `AmdNativeScore` and `AmdNativeSuiteReport` sidecars (`src/sol_execbench/core/scoring/amd_score.py:159`, `src/sol_execbench/core/scoring/amd_score.py:241`).

**State Management:**
- Runtime state is local to the CLI invocation, staging directory, subprocess, and generated sidecar files.
- `ProblemPackager.__del__()` removes the staging directory unless `--keep-staging` is set (`src/sol_execbench/driver/problem_packager.py:123`).
- No database or server process exists; persistent state is JSON/JSONL files under user-selected outputs, `data/`, `.planning/`, or generated artifacts.

## Key Abstractions

**Definition:**
- Purpose: Schema for the operation, symbolic axes, input/output tensors, and reference implementation.
- Examples: `src/sol_execbench/core/data/definition.py`, `examples/hip_cpp/rmsnorm/definition.json`.
- Pattern: Pydantic model with validators that parse reference code and enforce a top-level `run()` signature.

**Solution:**
- Purpose: Schema for user implementation metadata, sources, language category, target hardware, entry point, and compile options.
- Examples: `src/sol_execbench/core/data/solution.py`, `examples/hip_cpp/rmsnorm/solution_hip.json`.
- Pattern: Pydantic model with ROCm-only language validation, source path validation, and compile option validation.

**Workload:**
- Purpose: Concrete axis values, input generation descriptors, tolerance, and UUID for one executable case.
- Examples: `src/sol_execbench/core/data/workload.py`, `examples/hip_cpp/rmsnorm/workload.jsonl`.
- Pattern: Pydantic model with discriminated input descriptor unions and custom-input consistency checks.

**Trace:**
- Purpose: Canonical output record linking a definition, workload, solution, and evaluation result.
- Examples: `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/driver/templates/eval_driver.py`.
- Pattern: Pydantic model with status-dependent correctness/performance validation.

**BenchmarkConfig:**
- Purpose: Runtime knobs for warmup, iterations, clock lock requirement, reference timing, and seed.
- Examples: `src/sol_execbench/core/bench/config/benchmark_config.py`, `src/sol_execbench/core/bench/config/device_config.py`.
- Pattern: Dataclass-style config re-exported through `src/sol_execbench/core/__init__.py`.

**ProblemPackager:**
- Purpose: Bridge validated domain objects to a staging filesystem contract consumed by generated scripts.
- Examples: `src/sol_execbench/driver/problem_packager.py`, `tests/sol_execbench/driver/test_problem_packager.py`.
- Pattern: Stateful packager object that writes files in `__init__()`, provides compile/execute command builders, and parses JSONL stdout.

**Sidecar Evidence Models:**
- Purpose: Keep environment, profiler, static kernel, dependency, and score evidence versioned outside canonical trace objects.
- Examples: `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/scoring/amd_score.py`.
- Pattern: Versioned Pydantic/dataclass artifacts serialized as JSON sidecars.

## Entry Points

**Primary CLI:**
- Location: `src/sol_execbench/cli/main.py`
- Triggers: `uv run sol-execbench ...`, package console script `sol-execbench`.
- Responsibilities: Evaluate a solution, emit traces, and expose `contract`, `doctor`, and `toolchain` subcommands.

**Baseline CLI:**
- Location: `src/sol_execbench/cli/baseline.py`
- Triggers: `uv run sol-execbench-baseline ...`, package console script `sol-execbench-baseline`.
- Responsibilities: Compare candidate traces with baseline traces and output text or JSON summaries.

**Generated Build Driver:**
- Location: `src/sol_execbench/driver/templates/build_ext.py`
- Triggers: `ProblemPackager.compile()` writes the file; CLI runs `python build_ext.py`.
- Responsibilities: Compile staged native ROCm sources into `benchmark_kernel.so`.

**Generated Evaluation Driver:**
- Location: `src/sol_execbench/driver/templates/eval_driver.py`
- Triggers: `ProblemPackager.execute()` writes the file; CLI runs `python eval_driver.py`.
- Responsibilities: Execute reference/user code, validate correctness, time kernels, and emit `Trace` JSONL.

**Dataset Runner:**
- Location: `scripts/run_dataset.py`
- Triggers: `uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5`.
- Responsibilities: Run many benchmark problems through the CLI and assemble reports/sidecars.

**Dataset Downloader/Inspector:**
- Location: `scripts/download_solexecbench.py`, `scripts/inspect_dataset.py`, `scripts/report_parity_gaps.py`
- Triggers: Direct script invocation through `uv run`.
- Responsibilities: Acquire, inspect, and report dataset readiness without changing evaluator runtime.

**Docker Runtime:**
- Location: `scripts/run_docker.sh`, `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`.
- Triggers: `./scripts/run_docker.sh --build`.
- Responsibilities: Build/run ROCm-capable environment and expose GPU device nodes to tests/evaluation.

## Architectural Constraints

- **Threading:** The CLI is synchronous; generated evaluation drivers use the main process for evaluation and inspect thread counts to detect injected background work (`src/sol_execbench/driver/templates/eval_driver.py:551`, `src/sol_execbench/driver/templates/eval_driver.py:651`).
- **Subprocess boundary:** Compilation and evaluation must happen through staged subprocess commands, not inline in the CLI (`src/sol_execbench/cli/main.py:655`, `src/sol_execbench/cli/main.py:721`).
- **Global state:** `eval_driver.py` intentionally uses module-level variables after loading staged JSON files; keep this template self-contained because it is copied into isolated directories (`src/sol_execbench/driver/templates/eval_driver.py:47`, `src/sol_execbench/driver/templates/eval_driver.py:151`).
- **Canonical schema boundary:** Do not add optional environment, profiler, static evidence, or AMD score fields directly to `Trace`; use sidecars and reporting modules (`src/sol_execbench/core/environment.py:4`, `src/sol_execbench/core/reporting.py`).
- **ROCm-only solution schema:** New native solution categories must use ROCm names and validation paths; legacy CUDA/NVIDIA schema values are rejected in `BuildSpec` (`src/sol_execbench/core/data/solution.py:158`, `src/sol_execbench/core/data/solution.py:183`).
- **PyTorch compatibility namespace:** `torch.cuda` APIs remain the compatibility surface for PyTorch ROCm events and device checks; wrap or document ROCm semantics rather than renaming public PyTorch calls locally (`src/sol_execbench/core/bench/timing.py:76`, `tests/sol_execbench/test_rocm_migration_residue_audit.py`).
- **Circular imports:** `src/sol_execbench/core/__init__.py` re-exports many domain classes and is imported by generated templates; avoid making lower-level modules depend on CLI or driver modules.

## Anti-Patterns

### Mutating Canonical Trace Schema For Evidence

**What happens:** Optional runtime or scoring metadata is added to `Trace` or `Evaluation`.
**Why it's wrong:** The public trace schema is the benchmark output contract; optional evidence must not alter compatibility.
**Do this instead:** Add a versioned sidecar model in `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, or `src/sol_execbench/core/scoring/`, then write it from CLI/script helpers.

### Inline User Kernel Execution In The CLI

**What happens:** New code imports user solution sources directly in `src/sol_execbench/cli/main.py`.
**Why it's wrong:** The CLI owns orchestration; generated staging drivers own user-code imports, stdout isolation, reward-hack checks, and subprocess failure boundaries.
**Do this instead:** Extend `ProblemPackager` or `src/sol_execbench/driver/templates/eval_driver.py`, then call it through `_run_evaluation_command()` in `src/sol_execbench/cli/main.py`.

### Bypassing Pydantic Schemas With Raw Dicts

**What happens:** Core code passes unchecked JSON dicts across module boundaries.
**Why it's wrong:** Validators encode security constraints, ROCm-only schema rules, reference function checks, and trace status invariants.
**Do this instead:** Construct `Definition`, `Solution`, `Workload`, and `Trace` objects from `src/sol_execbench/core/data/` as early as possible at file boundaries.

### Treating Derived AMD Scores As Canonical Benchmark Output

**What happens:** Derived AMD-native reports are presented as the same artifact as trace JSONL.
**Why it's wrong:** Score reports are guarded, derived evidence with claim-level warnings and incomplete-evidence handling.
**Do this instead:** Keep canonical trace output in JSONL and build derived sidecars through `src/sol_execbench/core/scoring/amd_score.py` and `scripts/run_dataset.py`.

## Error Handling

**Strategy:** Validate user-facing inputs early with schema models and Click exceptions, isolate runtime failures in subprocesses, and convert per-workload execution failures into `Trace` objects when possible.

**Patterns:**
- Use `click.ClickException` for CLI argument/path problems in `src/sol_execbench/cli/main.py`.
- Use Pydantic validators for schema invariants in `src/sol_execbench/core/data/`.
- Use subprocess return codes for compile/evaluation process failures in `src/sol_execbench/cli/main.py`.
- Emit failed workload traces with `EvaluationStatus` for runtime, numerical, dtype, shape, invalid reference, timeout, compile, and reward-hack outcomes in `src/sol_execbench/core/data/trace.py`.
- Bound external diagnostics with timeouts and injectable runners in `src/sol_execbench/core/environment.py`.
- Treat optional evidence collection failures as sidecar warnings or skipped evidence, not canonical evaluation failures.

## Cross-Cutting Concerns

**Logging:** CLI output uses Rich console for human-readable progress and tables in `src/sol_execbench/cli/main.py`; generated drivers redirect noisy stdout to stderr and reserve original stdout for strict JSONL in `src/sol_execbench/driver/templates/eval_driver.py`.

**Validation:** Pydantic models in `src/sol_execbench/core/data/` validate benchmark objects; tests in `tests/sol_execbench/` assert contract stability, ROCm residue classifications, and schema/build behavior.

**Authentication:** Not applicable; this is a local CLI/library benchmark harness with no user identity layer.

**Security:** Source paths reject absolute paths and parent traversal in `src/sol_execbench/core/data/solution.py`; generated evaluation runs static source review and reward-hack checks in `src/sol_execbench/core/bench/reward_hack.py` and `src/sol_execbench/driver/templates/eval_driver.py`; repository policy forbids credentials and proprietary kernels.

**Hardware Claims:** ROCm hardware, dependency, and score claims are represented through guardrails and sidecar artifacts in `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/runtime_evidence.py`, and `src/sol_execbench/core/scoring/`.

---

*Architecture analysis: 2026-05-28*
