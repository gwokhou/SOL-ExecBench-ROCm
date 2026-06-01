<!-- refreshed: 2026-06-01 -->
# Architecture

**Analysis Date:** 2026-06-01

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    User And Automation Layer                 │
├──────────────────┬──────────────────┬───────────────────────┤
│ Evaluator CLI    │ Baseline CLI     │ Dataset/report scripts │
│ `src/sol_execbench/cli/main.py` │ `src/sol_execbench/cli/baseline.py` │ `scripts/` │
└────────┬─────────┴────────┬─────────┴──────────┬────────────┘
         │                  │                     │
         ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Typed Contracts And Core Services            │
│ `src/sol_execbench/core/data/`, `src/sol_execbench/core/bench/` │
│ `src/sol_execbench/core/dataset/`, `src/sol_execbench/core/scoring/` │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Staged Execution Boundary                   │
│ `src/sol_execbench/driver/problem_packager.py`               │
│ `src/sol_execbench/driver/templates/build_ext.py`            │
│ `src/sol_execbench/driver/templates/eval_driver.py`          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              ROCm Runtime, Native Artifacts, Evidence         │
│ `/tmp/sol_execbench_*`, `benchmark_kernel.so`, JSON sidecars  │
└─────────────────────────────────────────────────────────────┘
```

SOL ExecBench is a layered Python package. The evaluator loads schema-validated benchmark artifacts, stages the problem and submitted sources into a temporary directory, optionally compiles native HIP/C++ code, runs a generated evaluation driver in a subprocess, and parses canonical trace JSONL. Dataset and reporting workflows reuse the same contract, execution, and scoring services rather than duplicating evaluator logic.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `SolExecbenchCli` | Dispatch root evaluation calls and `contract`, `doctor`, and `toolchain` metadata subcommands. | `src/sol_execbench/cli/main.py` |
| `_evaluate_cli` | Load problem files, create staging, compile native solutions, run evaluation subprocesses, parse traces, write sidecars, set process exit status. | `src/sol_execbench/cli/main.py` |
| `ProblemPackager` | Materialize definition, workload, solution, config, source files, and generated driver templates into a staging directory. | `src/sol_execbench/driver/problem_packager.py` |
| `build_ext.py` template | Compile native ROCm sources with `torch.utils.cpp_extension.load` and normalize the output to `benchmark_kernel.so`. | `src/sol_execbench/driver/templates/build_ext.py` |
| `eval_driver.py` template | Import reference and user code, generate inputs, enforce reward-hack checks, measure correctness and latency, emit JSONL `Trace` objects. | `src/sol_execbench/driver/templates/eval_driver.py` |
| Data contracts | Own public schemas for definitions, workloads, solutions, traces, evaluator contracts, shapes, dtypes, and JSON helpers. | `src/sol_execbench/core/data/` |
| Benchmark services | Own staged runtime imports, input/output allocation, correctness, timing, clock locks, profiler evidence, and static kernel evidence. | `src/sol_execbench/core/bench/` |
| Dataset services | Own dataset discovery, manifests, inventory, readiness, ready subsets, run state, execution closure, paper denominators, parity gaps, and evidence refs. | `src/sol_execbench/core/dataset/` |
| Scoring services | Own AMD hardware models, bound graphs, SOL bound artifacts, SOLAR derivation, AMD-native score reports, and score guardrails. | `src/sol_execbench/core/scoring/` |
| Environment/toolchain services | Own ROCm probes, toolchain routing, compatibility matrices, Docker targets, dependency policy, and runtime evidence reports. | `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/compatibility.py`, `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/runtime_evidence.py` |
| Reporting services | Own trace summaries, baseline comparison, consistency, stability, claim upgrade, trust summary, and matrix diff reports. | `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/consistency.py`, `src/sol_execbench/core/evaluation_stability.py`, `src/sol_execbench/core/claim_upgrade.py`, `src/sol_execbench/core/trust_summary.py`, `src/sol_execbench/core/matrix_diff.py` |
| Batch runner | Discover dataset problems, wrap reference/custom solutions, invoke `sol-execbench`, persist traces, closure records, timing evidence, and derived score sidecars. | `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/runner.py` |

## Pattern Overview

**Overall:** Layered CLI with a generated, subprocess-isolated evaluator.

**Key Characteristics:**
- Treat Pydantic contracts in `src/sol_execbench/core/data/` as the authority for external JSON inputs and canonical trace outputs.
- Keep submitted solution code outside the CLI process through `ProblemPackager` and generated templates in `src/sol_execbench/driver/templates/`.
- Keep canonical benchmark results in trace JSONL; write profiling, static kernel evidence, environment snapshots, closure reports, and derived scores as sidecars.
- Use small service modules in `src/sol_execbench/core/` for report generation and ROCm diagnostics instead of extending trace schemas.
- Use scripts in `scripts/` as orchestration shells around importable package helpers in `src/sol_execbench/core/dataset/`.

## Layers

**Command Layer:**
- Purpose: Parse command-line options, load user-provided files, call core services, run subprocesses, and render output.
- Location: `src/sol_execbench/cli/`, `scripts/`
- Contains: Click commands, dataset runner, report CLIs, download helpers, Docker helper shell script.
- Depends on: `src/sol_execbench/core/`, `src/sol_execbench/driver/`, Click, Rich, `subprocess`.
- Used by: Console scripts in `pyproject.toml`, direct script invocations, tests in `tests/sol_execbench/` and `tests/examples/`.

**Contract Layer:**
- Purpose: Validate and serialize benchmark definitions, workloads, solutions, trace rows, and evaluator metadata.
- Location: `src/sol_execbench/core/data/`, `src/sol_execbench/core/bench/config/`
- Contains: `Definition`, `Workload`, `Solution`, `BuildSpec`, `Trace`, `EvaluationStatus`, `BenchmarkConfig`.
- Depends on: Pydantic, Python standard library, Torch dtype adapters in `src/sol_execbench/core/data/dtypes.py`.
- Used by: CLI loading, staging, generated evaluator, dataset inventory, scoring, tests, examples.

**Execution Layer:**
- Purpose: Stage trusted inputs and untrusted solution sources, compile native ROCm code, and evaluate workloads in a subprocess.
- Location: `src/sol_execbench/driver/`, `src/sol_execbench/core/bench/`
- Contains: `ProblemPackager`, generated `build_ext.py`, generated `eval_driver.py`, runtime import helpers, input generation, timing, correctness, reward-hack checks.
- Depends on: Torch ROCm, Triton/PyTorch user code, HIP/C++ extension builds, safetensors, ROCm runtime tools.
- Used by: `src/sol_execbench/cli/main.py` and dataset subprocess calls from `src/sol_execbench/core/dataset/runner.py`.

**Evidence And Diagnostics Layer:**
- Purpose: Collect non-canonical evidence without changing benchmark trace semantics.
- Location: `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/runtime_evidence.py`
- Contains: ROCm environment snapshots, toolchain routing decisions, `rocprofv3` profile metadata, static kernel artifacts, runtime evidence reports.
- Depends on: Bounded subprocess probes, ROCm tools, injectable runners for tests.
- Used by: CLI sidecars, `sol-execbench doctor`, `sol-execbench toolchain`, dataset timing evidence, report tests.

**Dataset Layer:**
- Purpose: Convert local benchmark trees into inventory, readiness, execution, closure, and derived evidence artifacts.
- Location: `src/sol_execbench/core/dataset/`
- Contains: Categories, checksums, layout, manifest, inventory, readiness, ready subsets, runner helpers, run state, closure records, parity and denominator reports.
- Depends on: Data contracts, scoring services, JSON files under dataset problem directories.
- Used by: `scripts/run_dataset.py`, `scripts/inspect_dataset.py`, report scripts, tests under `tests/sol_execbench/`.

**Scoring Layer:**
- Purpose: Build derived AMD-native score artifacts from canonical traces and independently derived SOL evidence.
- Location: `src/sol_execbench/core/scoring/`, `src/sol_execbench/sol_score.py`, `src/sol_execbench/data/amd_hardware_models/`
- Contains: Hardware models, bound graph and estimates, SOL v1/v2 artifacts, SOLAR derivation, AMD-native score report models.
- Depends on: `Trace`, `Definition`, `Workload`, packaged AMD hardware model data.
- Used by: Dataset runner, score report scripts, tests for AMD scoring and derivation.

## Data Flow

### Primary Request Path

1. CLI dispatch enters `SolExecbenchCli.main` and `_evaluate_cli` (`src/sol_execbench/cli/main.py:853`, `src/sol_execbench/cli/main.py:568`).
2. `_evaluate_cli` resolves `definition.json`, `workload.jsonl`, `solution.json`, and optional config into `Definition`, `Workload`, `Solution`, and `BenchmarkConfig` (`src/sol_execbench/cli/main.py:568`).
3. `ProblemPackager` writes normalized JSON and source files into `/tmp/sol_execbench_*` (`src/sol_execbench/driver/problem_packager.py:91`, `src/sol_execbench/driver/problem_packager.py:182`).
4. Native ROCm solutions compile through the generated `build_ext.py` command and produce `benchmark_kernel.so` (`src/sol_execbench/driver/problem_packager.py:189`, `src/sol_execbench/driver/templates/build_ext.py`).
5. `ProblemPackager.execute` writes the generated evaluator and returns `python eval_driver.py` (`src/sol_execbench/driver/problem_packager.py:217`).
6. `_run_evaluation_command` executes the generated evaluator in the staging directory with ROCm allocator settings (`src/sol_execbench/cli/main.py:444`).
7. `eval_driver.py` loads staged JSON, imports reference and user functions, reviews sources, executes each workload, measures correctness and latency, and emits one JSON `Trace` per workload (`src/sol_execbench/driver/templates/eval_driver.py:94`, `src/sol_execbench/driver/templates/eval_driver.py:122`, `src/sol_execbench/driver/templates/eval_driver.py:148`, `src/sol_execbench/driver/templates/eval_driver.py:171`, `src/sol_execbench/driver/templates/eval_driver.py:294`, `src/sol_execbench/driver/templates/eval_driver.py:555`).
8. The CLI parses stdout back into `Trace` models, writes optional output and evidence sidecars, prints JSON or Rich tables, and exits nonzero unless every trace passes (`src/sol_execbench/driver/problem_packager.py:242`, `src/sol_execbench/cli/main.py:568`).

### Native HIP/C++ Compile Flow

1. `Solution.spec.languages` identifies native ROCm categories (`hip_cpp`, `hipblas`, `miopen`, `ck`, `rocwmma`) through `ProblemPackager._is_cpp` (`src/sol_execbench/driver/problem_packager.py:91`).
2. `ProblemPackager.compile` injects ROCm offload architecture flags when needed and writes `build_ext.py` (`src/sol_execbench/driver/problem_packager.py:189`).
3. `build_ext.py` reads `solution.json`, collects `.hip`, `.cpp`, `.cc`, `.cxx`, or `.c` files from the staging directory, sets `PYTORCH_ROCM_ARCH` from target hardware, and calls `torch.utils.cpp_extension.load` (`src/sol_execbench/driver/templates/build_ext.py`).
4. `eval_driver.py` imports `benchmark_kernel.so` for native ROCm solutions through `load_user_function` (`src/sol_execbench/core/bench/eval_runtime.py`).

### Dataset Batch Flow

1. `scripts/run_dataset.py` detects either one problem directory or a dataset root with category subdirectories (`scripts/run_dataset.py:418`).
2. Dataset runner helpers build reference or custom solution JSON, construct the `sol-execbench` command, and invoke it in a subprocess (`src/sol_execbench/core/dataset/runner.py:81`, `src/sol_execbench/core/dataset/runner.py:153`, `src/sol_execbench/core/dataset/runner.py:186`).
3. Per-problem traces are written under the requested output directory and summarized with `inspect_traces` (`src/sol_execbench/core/dataset/runner.py`).
4. Optional AMD score, SOL bound, SOLAR derivation, timing evidence, and execution closure sidecars are generated from canonical traces (`src/sol_execbench/core/dataset/runner.py:311`, `src/sol_execbench/core/dataset/runner.py:452`, `scripts/run_dataset.py:418`).

### Schema And Contract Flow

1. `Definition` validates reference code, input/output naming, axis references, and shape resolution (`src/sol_execbench/core/data/definition.py:136`).
2. `Solution` validates languages, entry points, target hardware, source paths, and immutable content hashes (`src/sol_execbench/core/data/solution.py:265`).
3. `Workload` validates concrete axes, input kinds, UUIDs, and tolerance rules (`src/sol_execbench/core/data/workload.py:102`).
4. `Trace` and `Evaluation` enforce status-dependent correctness and performance fields (`src/sol_execbench/core/data/trace.py:113`, `src/sol_execbench/core/data/trace.py:176`).

**State Management:**
- Runtime benchmark state is local to CLI calls, staging directories, subprocesses, and sidecar files.
- `ProblemPackager` owns staging cleanup unless `--keep-staging` is set (`src/sol_execbench/driver/problem_packager.py`).
- `Solution` is frozen and memoizes a deterministic source hash in `_hash_cache` (`src/sol_execbench/core/data/solution.py:265`).
- Module-level constants define schema versions, tool IDs, static review rules, and category defaults across `src/sol_execbench/core/`.

## Key Abstractions

**Benchmark Contracts:**
- Purpose: Represent all external benchmark inputs and canonical outputs.
- Examples: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/trace.py`.
- Pattern: Pydantic models with validators and JSON serialization.

**`ProblemPackager`:**
- Purpose: Convert typed contracts into a runnable staging directory and hide generated-file details from the CLI.
- Examples: `src/sol_execbench/driver/problem_packager.py`.
- Pattern: Context-manageable staging object returning subprocess commands.

**Generated Evaluator Templates:**
- Purpose: Keep user code loading, correctness checks, timing, and reward-hack defenses in a subprocess-local script.
- Examples: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`.
- Pattern: Template files copied verbatim into staging directories.

**Sidecar Evidence Models:**
- Purpose: Attach diagnostic, scoring, and closure evidence without mutating canonical traces.
- Examples: `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/dataset/execution_closure.py`.
- Pattern: Versioned Pydantic/dataclass payloads serialized as deterministic JSON.

**Dataset Run State:**
- Purpose: Track selected problems, workloads, trace statuses, evidence requirements, and closure outcomes.
- Examples: `src/sol_execbench/core/dataset/run_state.py`, `src/sol_execbench/core/dataset/run_closure.py`, `scripts/run_dataset.py`.
- Pattern: Pure helper functions plus script-level orchestration.

**AMD-Native Scoring:**
- Purpose: Score passed traces against AMD SOL bounds and baselines while preserving claim boundaries.
- Examples: `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/sol_score.py`.
- Pattern: Derived reports built from canonical `Trace` objects and sidecar references.

## Entry Points

**Evaluator CLI:**
- Location: `src/sol_execbench/cli/main.py`
- Triggers: `sol-execbench` console script in `pyproject.toml`.
- Responsibilities: Evaluate one problem, print evaluator contract, print ROCm diagnostics, print toolchain routing diagnostics.

**Baseline CLI:**
- Location: `src/sol_execbench/cli/baseline.py`
- Triggers: `sol-execbench-baseline` console script in `pyproject.toml`.
- Responsibilities: Compare trace outputs against baseline artifacts.

**Dataset Runner:**
- Location: `scripts/run_dataset.py`
- Triggers: `uv run scripts/run_dataset.py ...`.
- Responsibilities: Run one problem or many dataset categories through the evaluator CLI and write summaries, traces, closure, timing, and score artifacts.

**Report Scripts:**
- Location: `scripts/report_amd_bound_sanity.py`, `scripts/report_claim_upgrade.py`, `scripts/report_consistency.py`, `scripts/report_evaluation_stability.py`, `scripts/report_paper_denominator.py`, `scripts/report_parity_gaps.py`, `scripts/report_trust_summary.py`, `scripts/diff_matrix_reports.py`, `scripts/export_matrix_schema.py`, `scripts/inspect_dataset.py`.
- Triggers: Direct `uv run scripts/<name>.py ...` invocations.
- Responsibilities: Build focused JSON/Markdown reports from core service modules.

**Docker Environment:**
- Location: `scripts/run_docker.sh`, `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`.
- Triggers: `./scripts/run_docker.sh --build`.
- Responsibilities: Build and enter a ROCm-capable container with device access and dependency checks.

## Architectural Constraints

- **Threading:** The CLI and dataset runner execute serial Python control flow; the generated evaluator checks thread counts around user execution to detect thread-injection reward hacks in `src/sol_execbench/driver/templates/eval_driver.py`.
- **Subprocess isolation:** User solutions execute in generated staging subprocesses; native builds run in staging with `build_ext.py`; dataset runs invoke the CLI as a subprocess through `src/sol_execbench/core/dataset/runner.py`.
- **GPU API naming:** PyTorch exposes ROCm through `torch.cuda` APIs; timing uses HIP-backed PyTorch events in `src/sol_execbench/core/bench/timing.py`.
- **Global state:** `console` in `src/sol_execbench/cli/main.py`, `_ELAPSED_TIME_ADDR` and static rule tables in `src/sol_execbench/core/bench/reward_hack.py`, and schema/version constants in `src/sol_execbench/core/` modules are module-level state.
- **Circular imports:** No circular import chain is apparent in the inspected paths; core convenience exports in `src/sol_execbench/core/__init__.py` stay limited to stable public models and services.
- **Canonical output boundary:** `Trace` JSONL is the benchmark output authority; environment snapshots, static evidence, profiling, AMD scores, and closure reports are derived sidecars.
- **ROCm-only schema:** `BuildSpec` rejects legacy CUDA/NVIDIA language and compile-option values in `src/sol_execbench/core/data/solution.py`.

## Anti-Patterns

### Running Submitted Code In CLI Process

**What happens:** Importing submitted source directly from `src/sol_execbench/cli/main.py` bypasses the staging boundary.
**Why it's wrong:** It weakens subprocess isolation, reward-hack checks, stdout redirection, and staging cleanup.
**Do this instead:** Use `ProblemPackager.execute` and the generated `eval_driver.py` flow in `src/sol_execbench/driver/problem_packager.py` and `src/sol_execbench/driver/templates/eval_driver.py`.

### Extending Canonical Trace For Optional Evidence

**What happens:** Adding profiler, environment, static-evidence, or score fields to `Trace` changes public benchmark semantics.
**Why it's wrong:** `Trace` validation in `src/sol_execbench/core/data/trace.py` is the canonical output contract.
**Do this instead:** Add versioned sidecar payloads through modules such as `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, or `src/sol_execbench/core/scoring/amd_score.py`.

### Duplicating Dataset CLI Invocation Logic

**What happens:** Scripts construct evaluator commands or parse trace output independently.
**Why it's wrong:** It drifts from timeout, logging, and output parsing behavior in the shared dataset runner.
**Do this instead:** Use `build_cli_command`, `run_cli`, and reporting helpers in `src/sol_execbench/core/dataset/runner.py`.

### Mixing Python And Native Solution Languages

**What happens:** A solution spec includes both Python categories (`pytorch`, `triton`) and native ROCm categories (`hip_cpp`, `hipblas`, `miopen`, `ck`, `rocwmma`).
**Why it's wrong:** `BuildSpec` enforces mutually exclusive build and import paths.
**Do this instead:** Use one solution category group per `Solution` in `src/sol_execbench/core/data/solution.py`.

## Error Handling

**Strategy:** Convert invalid user inputs to validation exceptions or Click exceptions at the boundary, convert runtime failures to trace statuses inside the generated evaluator, and keep optional evidence failures nonfatal.

**Patterns:**
- CLI file-resolution failures raise `click.ClickException` in `src/sol_execbench/cli/main.py`.
- Pydantic validators reject invalid contracts in `src/sol_execbench/core/data/`.
- Generated evaluator failures emit `Trace` rows with `INVALID_REFERENCE`, `INCORRECT_SHAPE`, `INCORRECT_DTYPE`, `INCORRECT_NUMERICAL`, `RUNTIME_ERROR`, or `REWARD_HACK` in `src/sol_execbench/driver/templates/eval_driver.py`.
- Optional environment, profiler, and static-evidence collection catches exceptions and writes warnings or skipped metadata in `src/sol_execbench/cli/main.py`.
- Dataset subprocess failures write bounded CLI logs through `src/sol_execbench/core/dataset/runner.py`.

## Cross-Cutting Concerns

**Logging:** CLI user output uses Rich in `src/sol_execbench/cli/main.py`; dataset scripts use stdout/stderr and bounded log files in `src/sol_execbench/core/dataset/runner.py`.
**Validation:** Pydantic models validate public JSON schemas in `src/sol_execbench/core/data/`, `src/sol_execbench/core/dataset/`, `src/sol_execbench/core/scoring/`, `src/sol_execbench/core/environment.py`, and `src/sol_execbench/core/toolchain.py`.
**Authentication:** Not applicable; local CLI and dataset tooling do not include an authentication layer.
**Security:** Source paths reject absolute paths and parent traversal in `src/sol_execbench/core/data/solution.py`; reward-hack static and runtime checks live in `src/sol_execbench/core/bench/reward_hack.py`; secrets and downloaded datasets stay outside committed outputs.
**Reproducibility:** `BenchmarkConfig` controls seed, warmup, iterations, reference timing, and clock-lock policy in `src/sol_execbench/core/bench/config/benchmark_config.py`.
**ROCm compatibility:** Environment, toolchain, Docker, dependency, and compatibility services live in `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, and `src/sol_execbench/core/compatibility.py`.

---

*Architecture analysis: 2026-06-01*
