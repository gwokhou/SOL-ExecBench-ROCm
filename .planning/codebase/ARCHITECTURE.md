<!-- refreshed: 2026-06-04 -->
# Architecture

**Analysis Date:** 2026-06-04

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    User and Batch Entrypoints                │
├──────────────────┬──────────────────┬───────────────────────┤
│  Click evaluator │  Baseline CLI    │ Dataset/release tools │
│`src/sol_execbench/cli/main.py`│`src/sol_execbench/cli/baseline.py`│`scripts/run_dataset.py`│
└────────┬─────────┴────────┬─────────┴──────────┬────────────┘
         │                  │                     │
         ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Domain Models and Core Services             │
│ `src/sol_execbench/core/data/`, `src/sol_execbench/core/*`    │
├───────────────────────┬───────────────────────┬─────────────┤
│ Benchmark Runtime     │ Dataset Utilities      │ Scoring     │
│ `src/sol_execbench/core/bench/` │ `src/sol_execbench/core/dataset/` │ `src/sol_execbench/core/scoring/` │
└────────┬──────────────┴───────────┬───────────┴──────┬──────┘
         │                          │                  │
         ▼                          ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                Staged Execution and Native Build             │
│ `src/sol_execbench/driver/problem_packager.py`               │
│ `src/sol_execbench/driver/templates/eval_driver.py`          │
│ `src/sol_execbench/driver/templates/build_ext.py`            │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│       Filesystem Artifacts, ROCm Tools, PyTorch HIP Events   │
│ `definition.json`, `workload.jsonl`, `solution.json`, traces │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Evaluator CLI | Parse command options, load problem files, stage execution, compile native solutions, run evaluation subprocesses, write trace and sidecar outputs | `src/sol_execbench/cli/main.py` |
| Baseline CLI | Compare trace JSONL files against baseline performance and guardrails | `src/sol_execbench/cli/baseline.py` |
| ProblemPackager | Materialize `Definition`, `Workload`, `Solution`, and `BenchmarkConfig` into a temporary staging directory and return compile/evaluate commands | `src/sol_execbench/driver/problem_packager.py` |
| Eval driver template | Run inside the staging directory, import user/reference code, generate inputs, enforce reward-hack defenses, measure correctness/performance, emit JSONL `Trace` objects | `src/sol_execbench/driver/templates/eval_driver.py` |
| Build template | Build HIP/C++ solutions into `benchmark_kernel.so` with `torch.utils.cpp_extension` | `src/sol_execbench/driver/templates/build_ext.py` |
| Data schemas | Validate public benchmark contracts for definitions, workloads, solutions, traces, dtypes, and JSON helpers | `src/sol_execbench/core/data/` |
| Benchmark helpers | Provide input generation, correctness checks, reward-hack checks, timing, clock locking, static evidence, and rocprofv3 profiling helpers | `src/sol_execbench/core/bench/` |
| Dataset helpers | Migrate source datasets, describe dataset layout, compute manifests/checksums, shard work, record run state, and support batch runners | `src/sol_execbench/core/dataset/` |
| Scoring helpers | Derive AMD SOL bounds, AMD-native scores, baselines, sanity reports, and solar derivation evidence | `src/sol_execbench/core/scoring/` |
| Environment/toolchain evidence | Probe ROCm/PyTorch/toolchain status and produce machine-readable diagnostics | `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py` |
| Packaged data | Store validated built-in AMD hardware model payloads | `src/sol_execbench/data/amd_hardware_models/` |
| Dataset runner script | Execute problem sets through the installed `sol-execbench` CLI, persist traces/logs/evidence refs, and build closure/scoring outputs | `scripts/run_dataset.py` |

## Pattern Overview

**Overall:** Layered Python package with schema-first domain models, command-line orchestration, and isolated staged execution.

**Key Characteristics:**
- Public payloads enter through Pydantic models in `src/sol_execbench/core/data/` before reaching evaluation logic.
- User solution code is copied into a temporary staging directory and executed in a subprocess through `src/sol_execbench/driver/templates/eval_driver.py`.
- Native ROCm solution categories are compiled before evaluation; Python/Triton solution categories are imported directly from staged source files.
- Core modules are importable services; scripts under `scripts/` orchestrate longer workflows by composing those services and the CLI.
- ROCm port compatibility keeps PyTorch's `torch.cuda` API as the device-event interface because PyTorch ROCm exposes HIP devices through that namespace.

## Layers

**CLI Layer:**
- Purpose: Convert command-line invocations into typed evaluation, metadata, diagnostics, migration, or baseline workflows.
- Location: `src/sol_execbench/cli/`
- Contains: Click commands, option parsing, terminal output, subprocess orchestration, sidecar writing.
- Depends on: `src/sol_execbench/core/`, `src/sol_execbench/driver/`, `click`, `rich`.
- Used by: `pyproject.toml` console scripts, `scripts/run_dataset.py`, user shells, tests under `tests/sol_execbench/`.
- Add new evaluator subcommands in `src/sol_execbench/cli/main.py` when they need the `sol-execbench` executable namespace; add separate console commands only when the workflow is independent like `src/sol_execbench/cli/baseline.py`.

**Schema Layer:**
- Purpose: Define and validate benchmark contracts for portable JSON/JSONL artifacts.
- Location: `src/sol_execbench/core/data/`
- Contains: `Definition`, `Workload`, `Solution`, `Trace`, `Evaluation`, dtype conversion, shape expression resolution, JSON helpers.
- Depends on: Pydantic, Python standard library, optional torch dtype mappings.
- Used by: Every CLI, driver, dataset, and scoring workflow.
- Use `BaseModelWithDocstrings` in `src/sol_execbench/core/data/base_model.py` for public contract models with field documentation.

**Benchmark Runtime Layer:**
- Purpose: Execute one solution against one or more workloads and produce canonical traces.
- Location: `src/sol_execbench/core/bench/`
- Contains: Benchmark configuration, input/output allocation, correctness checks, reward-hack checks, latency timing, clock lock checks, static kernel evidence, rocprofv3 profile collection.
- Depends on: PyTorch ROCm, staged model objects, filesystem sidecars, optional ROCm tools.
- Used by: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/cli/main.py`, tests under `tests/sol_execbench/core/bench/`.
- Keep GPU-sensitive logic here rather than in `src/sol_execbench/cli/main.py`.

**Driver/Staging Layer:**
- Purpose: Isolate user code, generated reference modules, native builds, and evaluation output in a bounded temporary directory.
- Location: `src/sol_execbench/driver/`
- Contains: `ProblemPackager`, `eval_driver.py` template, `build_ext.py` template.
- Depends on: Core schemas and benchmark helpers.
- Used by: Evaluator CLI and driver tests.
- Put code that writes staging files or changes compile/evaluate command construction in `src/sol_execbench/driver/problem_packager.py`; put code that runs inside the subprocess in `src/sol_execbench/driver/templates/eval_driver.py`.

**Dataset Layer:**
- Purpose: Convert downloaded benchmark sources into local layout, track manifests and checksums, run dataset batches, and close evidence loops.
- Location: `src/sol_execbench/core/dataset/`
- Contains: Category definitions, checksums, evidence refs, execution closure, inventory, layout, migration, readiness, run closure, run state, sharding, runner helpers.
- Depends on: Core schemas, scoring helpers, subprocess CLI execution.
- Used by: `src/sol_execbench/cli/main.py` dataset commands, `scripts/run_dataset.py`, dataset tests.
- Add reusable dataset behavior to `src/sol_execbench/core/dataset/`; keep command-specific wiring in `scripts/run_dataset.py` or CLI functions.

**Scoring Layer:**
- Purpose: Build derived AMD-native score artifacts from traces, baselines, hardware models, and SOL-bound evidence.
- Location: `src/sol_execbench/core/scoring/`
- Contains: AMD bound graphs, hardware models, AMD SOL v1/v2 artifacts, AMD-native score reports, baseline artifacts, solar derivation evidence, sanity reports.
- Depends on: Core schemas, `src/sol_execbench/sol_score.py`, packaged data in `src/sol_execbench/data/amd_hardware_models/`.
- Used by: Dataset runner, report scripts, release validation scripts, tests.
- Add new score artifact models and derivation rules here; keep output rendering in scripts or report modules when possible.

**Docs/Release Tooling Layer:**
- Purpose: Produce public documentation, release readiness reports, and repository validation artifacts.
- Location: `docs/`, `scripts/`
- Contains: User-facing docs, internal docs, release scripts, reporting scripts, dataset download/inspection tools.
- Depends on: Importable core helpers and repository files.
- Used by: Maintainers and tests that enforce docs/report contracts.

## Data Flow

### Primary Request Path

1. User invokes `sol-execbench` through the script configured in `pyproject.toml:28`.
2. `SolExecbenchCli` dispatches root evaluation or subcommands in `src/sol_execbench/cli/main.py:1033`.
3. `_evaluate_cli` resolves `definition.json`, `workload.jsonl`, `solution.json`, and optional `config.json` in `src/sol_execbench/cli/main.py:633`.
4. `_load_definition`, `_load_workloads`, `_load_solution`, and `_load_config` instantiate typed models in `src/sol_execbench/cli/main.py:98`.
5. `ProblemPackager` writes typed payloads and solution sources into a temporary staging directory in `src/sol_execbench/driver/problem_packager.py:92`.
6. HIP/C++ categories compile through `ProblemPackager.compile()` and generated `build_ext.py` in `src/sol_execbench/driver/problem_packager.py:179`.
7. `ProblemPackager.execute()` writes `eval_driver.py` and returns `python eval_driver.py` in `src/sol_execbench/driver/problem_packager.py:211`.
8. `src/sol_execbench/driver/templates/eval_driver.py` loads staged files, imports reference and user functions, evaluates workloads, and emits strict JSONL traces.
9. `ProblemPackager.convert_stdout_to_traces()` parses JSONL into `Trace` models in `src/sol_execbench/driver/problem_packager.py:232`.
10. The CLI writes optional trace JSONL, environment snapshot, profile sidecar, and static evidence sidecar in `src/sol_execbench/cli/main.py:633`.

### Dataset Batch Path

1. User invokes `uv run scripts/run_dataset.py <benchmark-root>` through `scripts/run_dataset.py`.
2. `discover_problems()` selects problem directories in `scripts/run_dataset.py:120`.
3. `build_solution_for_problem()` creates a reference or custom `Solution` JSON payload in `src/sol_execbench/core/dataset/runner.py:74`.
4. `build_cli_command()` points at the installed `sol-execbench` executable and requests JSON output in `src/sol_execbench/core/dataset/runner.py:133`.
5. `run_cli()` executes the CLI with stdout/stderr captured to files in `src/sol_execbench/core/dataset/runner.py:166`.
6. Dataset run state, closure records, derived evidence refs, and AMD-native scores are assembled by `scripts/run_dataset.py` using helpers from `src/sol_execbench/core/dataset/` and `src/sol_execbench/core/scoring/`.

### Contract and Diagnostics Path

1. `sol-execbench contract --json` dispatches to `_contract_cli` in `src/sol_execbench/cli/main.py:867`.
2. `build_evaluator_contract()` returns the GPU-free evaluator contract from `src/sol_execbench/core/data/contract.py`.
3. `sol-execbench doctor --json` dispatches to `_doctor_cli` in `src/sol_execbench/cli/main.py:878`.
4. `build_environment_diagnostics()` probes ROCm/PyTorch/device status in `src/sol_execbench/core/environment.py`.
5. `sol-execbench toolchain --json` dispatches to `_toolchain_cli` in `src/sol_execbench/cli/main.py:889`.
6. `build_toolchain_routing_report()` evaluates requested evidence/toolchain support in `src/sol_execbench/core/toolchain.py`.

### Dataset Migration Path

1. `sol-execbench dataset migrate-sol` dispatches to `_dataset_migrate_sol_cli` in `src/sol_execbench/cli/main.py:952`.
2. `migrate_sol_execbench()` converts downloaded SOL ExecBench inputs into local problem directories in `src/sol_execbench/core/dataset/migration.py`.
3. `write_migration_manifest()` writes migration metadata in `src/sol_execbench/core/dataset/manifest.py`.
4. `sol-execbench dataset migrate-flashinfer` uses `migrate_flashinfer_trace()` through `src/sol_execbench/cli/main.py:998`.

**State Management:**
- Evaluation state is filesystem-scoped: staged directories, JSON/JSONL problem files, compiled `benchmark_kernel.so`, stdout/stderr logs, traces, and sidecars.
- The runtime avoids long-lived application state. Module-level constants define schema versions, tool names, and policy values in files such as `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/scoring/amd_score.py`, and `src/sol_execbench/core/environment.py`.
- `ProblemPackager` owns temporary directory cleanup unless `keep_output_dir=True` is passed in `src/sol_execbench/driver/problem_packager.py`.

## Key Abstractions

**Definition:**
- Purpose: Public schema for a computational workload, symbolic axes, tensor specs, and Python reference code.
- Examples: `src/sol_execbench/core/data/definition.py`, `docs/definition.md`, `tests/sol_execbench/core/data/test_definition.py`.
- Pattern: Pydantic model with validators for reference syntax, input ordering, custom input entrypoints, and shape axes.

**Workload:**
- Purpose: Concrete axis/input/tolerance values for one benchmark case.
- Examples: `src/sol_execbench/core/data/workload.py`, `tests/sol_execbench/core/data/test_workload.py`.
- Pattern: Pydantic model with discriminated input specs for random, scalar, safetensors, and custom inputs.

**Solution:**
- Purpose: Public schema for user implementation metadata, source files, target hardware, language category, entry point, and native compile options.
- Examples: `src/sol_execbench/core/data/solution.py`, `docs/solution.md`, `examples/hip_cpp/`, `examples/triton/`.
- Pattern: Pydantic model with source-path boundary validation, compile-flag validation, entry-point validation, and stable hash generation.

**Trace:**
- Purpose: Canonical evaluation result linking a definition, solution, workload, status, correctness metrics, performance metrics, and environment data.
- Examples: `src/sol_execbench/core/data/trace.py`, `docs/trace.md`.
- Pattern: Pydantic model with status-dependent correctness/performance validation.

**BenchmarkConfig:**
- Purpose: Runtime configuration for warmup/rep counts, clock locking, and seed handling.
- Examples: `src/sol_execbench/core/bench/config/benchmark_config.py`, `src/sol_execbench/core/bench/config/device_config.py`.
- Pattern: Dataclass consumed by CLI, staging, and eval driver code.

**ProblemPackager:**
- Purpose: Convert in-memory typed objects into a self-contained directory that a subprocess can compile and evaluate.
- Examples: `src/sol_execbench/driver/problem_packager.py`, `tests/sol_execbench/driver/`.
- Pattern: Context-manager resource owner with explicit `compile()`, `execute()`, and `convert_stdout_to_traces()` phases.

**AMD Scoring Artifacts:**
- Purpose: Represent hardware models, SOL-bound estimates, solar derivation evidence, and AMD-native scores.
- Examples: `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/solar_derivation.py`.
- Pattern: Dataclasses and Pydantic models with `to_dict()` methods for stable JSON artifacts.

## Entry Points

**Evaluator CLI:**
- Location: `src/sol_execbench/cli/main.py`
- Triggers: `sol-execbench` console script in `pyproject.toml`.
- Responsibilities: Single-problem evaluation, contract output, environment diagnostics, toolchain routing, dataset migration.

**Baseline CLI:**
- Location: `src/sol_execbench/cli/baseline.py`
- Triggers: `sol-execbench-baseline` console script in `pyproject.toml`.
- Responsibilities: Load current/baseline traces, compare latency/status outcomes, emit text or JSON summaries.

**Dataset Runner:**
- Location: `scripts/run_dataset.py`
- Triggers: `uv run scripts/run_dataset.py <downloaded-benchmark-dir>`.
- Responsibilities: Discover dataset problems, construct solutions, invoke `sol-execbench`, collect traces/logs, build closure and score artifacts.

**Docker Runner:**
- Location: `scripts/run_docker.sh`
- Triggers: `./scripts/run_docker.sh --build`.
- Responsibilities: Build and enter ROCm-capable container environment with GPU device access and dependency preflight.

**Release and Report Scripts:**
- Location: `scripts/`
- Triggers: Direct `uv run scripts/*.py` invocations and tests.
- Responsibilities: Generate claim upgrade, consistency, trust, matrix, prerelease, and validation reports using core helpers.

**Generated Eval Driver:**
- Location: `src/sol_execbench/driver/templates/eval_driver.py`
- Triggers: Written to staging by `ProblemPackager.execute()` and run as `python eval_driver.py`.
- Responsibilities: Execute untrusted benchmark solution code in a subprocess boundary and emit canonical JSONL traces.

## Architectural Constraints

- **Threading:** Core evaluation is subprocess-based and mostly single-process per CLI call; dataset-scale execution can use `ThreadPoolExecutor` in `scripts/run_dataset.py`; eval driver checks thread injection with `src/sol_execbench/core/bench/reward_hack.py`.
- **Global state:** Eval driver intentionally mutates stdout file descriptors and `sys.path` in `src/sol_execbench/driver/templates/eval_driver.py` to keep library noise off JSON stdout and allow staged imports. Avoid importing this template as normal package code.
- **GPU API naming:** ROCm execution uses PyTorch's historical `torch.cuda` namespace for HIP devices in `src/sol_execbench/core/bench/timing.py` and `src/sol_execbench/driver/templates/eval_driver.py`.
- **Subprocess boundaries:** User code must run through staging and `subprocess.run()` from `src/sol_execbench/cli/main.py`; do not import user solutions directly into the CLI process.
- **Filesystem boundary:** `SourceFile` rejects absolute paths and `..`, while compile flags reject host path injection in `src/sol_execbench/core/data/solution.py`.
- **Secrets:** Do not read `.env`, credential, token, or key files. Environment diagnostics in `src/sol_execbench/core/environment.py` should report only controlled, non-secret environment details.
- **Public schemas:** Changes to `src/sol_execbench/core/data/` are public contract changes and require docs/tests updates in `docs/` and `tests/sol_execbench/core/data/`.

## Anti-Patterns

### CLI-Owned Benchmark Logic

**What happens:** Adding GPU timing, input generation, correctness, or reward-hack logic directly inside `src/sol_execbench/cli/main.py`.
**Why it's wrong:** The generated eval driver needs the same logic inside the isolated subprocess, and CLI-only logic bypasses existing tests under `tests/sol_execbench/core/bench/`.
**Do this instead:** Add runtime behavior to `src/sol_execbench/core/bench/` and call it from `src/sol_execbench/driver/templates/eval_driver.py` or the CLI sidecar orchestration.

### Direct User-Code Import In Core Processes

**What happens:** Importing solution source files from `src/sol_execbench/cli/main.py`, `scripts/run_dataset.py`, or scoring modules.
**Why it's wrong:** It bypasses staging, stdout isolation, reward-hack checks, and cleanup owned by `ProblemPackager`.
**Do this instead:** Build a `Solution` model, stage it with `src/sol_execbench/driver/problem_packager.py`, and run `eval_driver.py` in a subprocess.

### Ad Hoc JSON Payloads For Public Artifacts

**What happens:** Constructing trace, definition, solution, or workload dictionaries outside model constructors.
**Why it's wrong:** Public schemas enforce validation and status invariants in `src/sol_execbench/core/data/`.
**Do this instead:** Instantiate `Definition`, `Workload`, `Solution`, `Trace`, and related models, then serialize with `model_dump(mode="json")` or `model_dump_json()`.

### Mixing Dataset Script Logic Into Package Models

**What happens:** Adding command-specific output path, logging, or CLI argument behavior inside `src/sol_execbench/core/data/` or scoring models.
**Why it's wrong:** Core models are reused by CLIs, scripts, docs tests, and public APIs; command policy belongs at the orchestration edge.
**Do this instead:** Put reusable transformations in `src/sol_execbench/core/dataset/` or `src/sol_execbench/core/scoring/`, and keep script-specific file layout in `scripts/run_dataset.py`.

## Error Handling

**Strategy:** Validate public inputs early with Pydantic, keep subprocess failures diagnostic-rich, and encode per-workload evaluation failures as `Trace.evaluation.status` rather than Python exceptions whenever possible.

**Patterns:**
- Raise `click.ClickException` for user-facing CLI argument and mode errors in `src/sol_execbench/cli/main.py`.
- Use Pydantic validators for schema violations in `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/workload.py`, and `src/sol_execbench/core/data/trace.py`.
- Convert eval-driver workload failures to `Trace` objects with `EvaluationStatus.RUNTIME_ERROR`, `INCORRECT_*`, `INVALID_REFERENCE`, or `REWARD_HACK` in `src/sol_execbench/driver/templates/eval_driver.py`.
- Persist bounded no-trace diagnostics sidecars for subprocess failures that produce no parseable traces in `src/sol_execbench/cli/main.py`.
- Return explicit diagnostic dataclasses/models for environment and toolchain probes in `src/sol_execbench/core/environment.py` and `src/sol_execbench/core/toolchain.py`.

## Cross-Cutting Concerns

**Logging:** CLI uses `rich.console.Console(stderr=True)` in `src/sol_execbench/cli/main.py`. Eval driver redirects normal stdout to stderr and writes only JSONL traces to the saved real stdout in `src/sol_execbench/driver/templates/eval_driver.py`.

**Validation:** Pydantic schema validation lives in `src/sol_execbench/core/data/`; native compile flag and path boundary validation live in `src/sol_execbench/core/data/solution.py`; report payload validation is distributed across `src/sol_execbench/core/scoring/` and `src/sol_execbench/core/dataset/`.

**Authentication:** Not applicable. This is a local benchmark package. Do not add network authentication paths to core evaluation. Dataset download helpers should keep tokens and credentials outside repository files.

**Security:** Staging, source-path validation, compile-flag restrictions, static source review, critical-function integrity checks, lazy-output checks, thread-injection checks, and stdout isolation are core safety boundaries. Relevant files: `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/eval_driver.py`.

**ROCm Evidence:** Environment snapshots, rocprofv3 sidecars, static kernel evidence, toolchain routing, and AMD-native score warnings keep claims bounded to available evidence. Relevant files: `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/scoring/amd_score.py`.

---

*Architecture analysis: 2026-06-04*
