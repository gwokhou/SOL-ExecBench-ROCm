<!-- refreshed: 2026-05-24 -->
# Architecture

**Analysis Date:** 2026-05-24

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    Command And Script Layer                  │
├──────────────────┬──────────────────┬───────────────────────┤
│ Evaluator CLI    │ Baseline CLI     │ Dataset scripts       │
│ `src/sol_execbench/cli/main.py` │ `src/sol_execbench/cli/baseline.py` │ `scripts/run_dataset.py` │
└────────┬─────────┴──────────────────┴──────────┬────────────┘
         │                                        │
         ▼                                        ▼
┌─────────────────────────────────────────────────────────────┐
│             Typed Contracts, Benchmark Core, Scoring         │
│ `src/sol_execbench/core/data/`                               │
│ `src/sol_execbench/core/bench/`                              │
│ `src/sol_execbench/core/dataset/`                            │
│ `src/sol_execbench/core/scoring/`                            │
└────────┬────────────────────────────────────────┬────────────┘
         │                                        │
         ▼                                        ▼
┌─────────────────────────────────────────────────────────────┐
│             Staging, Generated Drivers, Sidecar Output       │
│ `src/sol_execbench/driver/problem_packager.py`               │
│ `src/sol_execbench/driver/templates/`                        │
│ `data/`, `examples/`, output directories                     │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Evaluator CLI | Load `definition.json`, `workload.jsonl`, `solution.json`, stage execution, run compile/eval subprocesses, render traces or JSONL. | `src/sol_execbench/cli/main.py` |
| Contract command | Print the GPU-free evaluator compatibility contract through the same root command dispatch. | `src/sol_execbench/cli/main.py` |
| Baseline CLI | Compare trace outputs against baseline artifacts and emit claim guardrail warnings. | `src/sol_execbench/cli/baseline.py` |
| Public model exports | Provide stable package imports for schema and config models. | `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/data/__init__.py` |
| Data contracts | Define Pydantic models for definitions, workloads, solutions, traces, evaluator contract metadata, dtypes, and shape resolution. | `src/sol_execbench/core/data/` |
| Benchmark runtime | Generate inputs, load safetensors, compute correctness, time GPU callables, enforce clock policy, and detect reward hacks. | `src/sol_execbench/core/bench/` |
| Problem staging | Write source and JSON inputs to a temporary working directory and return commands for compilation and evaluation. | `src/sol_execbench/driver/problem_packager.py` |
| Generated build driver | Compile native ROCm solutions through `torch.utils.cpp_extension.load` and produce `benchmark_kernel.so`. | `src/sol_execbench/driver/templates/build_ext.py` |
| Generated eval driver | Import reference and solution code, run correctness and timing, and emit one `Trace` JSON object per workload. | `src/sol_execbench/driver/templates/eval_driver.py` |
| Dataset sidecars | Inspect downloaded dataset layout and write deterministic manifest, inventory, readiness, ready-subset, and parity-gap artifacts. | `src/sol_execbench/core/dataset/` |
| Dataset runner | Discover problems, wrap reference/custom solutions, invoke the evaluator CLI, collect timing evidence, and write summaries/closure reports. | `scripts/run_dataset.py` |
| AMD scoring | Build derived AMD SOL bounds and guarded AMD-native score reports from canonical traces and sidecar evidence. | `src/sol_execbench/core/scoring/` |
| Reporting | Summarize trace collections and construct derived evidence reports without mutating trace schema. | `src/sol_execbench/core/reporting.py` |

## Pattern Overview

**Overall:** Layered Python package with typed data contracts, subprocess-isolated GPU execution, and deterministic sidecar reporting.

**Key Characteristics:**
- Keep schemas in `src/sol_execbench/core/data/` and pass validated Pydantic models across CLI, driver, dataset, and scoring boundaries.
- Keep GPU/user-code execution out of the CLI process. Use `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py` to create a staging directory, then run generated scripts from `src/sol_execbench/driver/templates/`.
- Preserve canonical benchmark output as `Trace` JSONL from `src/sol_execbench/core/data/trace.py`; put summaries, scores, inventories, and readiness data in separate derived sidecars.
- Treat ROCm as the active backend while retaining PyTorch's historical `torch.cuda` API surface where PyTorch ROCm exposes HIP-backed device behavior.

## Layers

**CLI And Script Layer:**
- Purpose: Convert user commands into validated model loads, subprocess calls, and human or JSON output.
- Location: `src/sol_execbench/cli/`, `scripts/`
- Contains: Click commands, argparse scripts, dataset orchestration, download/report helpers.
- Depends on: `src/sol_execbench/core/`, `src/sol_execbench/driver/`, Python subprocess APIs.
- Used by: Users, tests in `tests/sol_execbench/test_e2e.py`, dataset workflows, Docker workflows.

**Typed Contract Layer:**
- Purpose: Define every public benchmark schema and validation rule.
- Location: `src/sol_execbench/core/data/`
- Contains: `Definition`, `Workload`, `Solution`, `Trace`, `Evaluation`, contract metadata, dtype and shape helpers.
- Depends on: Pydantic, Python AST parsing, dtype mapping helpers.
- Used by: CLI loaders, generated drivers, dataset inventory/readiness, scoring, tests, examples.

**Benchmark Runtime Layer:**
- Purpose: Execute benchmark semantics inside the evaluation process.
- Location: `src/sol_execbench/core/bench/`
- Contains: Input generation, safetensors loading, correctness checks, timing, clock-lock policy, `rocprofv3` evidence helpers, reward-hack defenses.
- Depends on: PyTorch ROCm, safetensors, typed data models.
- Used by: `src/sol_execbench/driver/templates/eval_driver.py`, `scripts/run_dataset.py`, benchmark tests.

**Driver/Staging Layer:**
- Purpose: Materialize problem files and generated drivers into an isolated temporary directory.
- Location: `src/sol_execbench/driver/`
- Contains: `ProblemPackager`, generated `build_ext.py`, generated `eval_driver.py`.
- Depends on: Core data models, ROCm target detection tools, subprocess execution.
- Used by: `src/sol_execbench/cli/main.py`, tests under `tests/sol_execbench/driver/`.

**Dataset Intelligence Layer:**
- Purpose: Produce deterministic acquisition/layout/readiness artifacts for downloaded SOL ExecBench data.
- Location: `src/sol_execbench/core/dataset/`, `scripts/download_solexecbench.py`, `scripts/inspect_dataset.py`, `scripts/report_parity_gaps.py`
- Contains: Manifest, inventory, readiness, ready-subset, parity-gap models and writers.
- Depends on: Core data models, checksum helpers, local filesystem layout.
- Used by: Dataset scripts, release/validation tests, planning docs.

**Scoring And Reporting Layer:**
- Purpose: Convert traces and derived evidence into AMD-native scores and summaries.
- Location: `src/sol_execbench/core/scoring/`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/sol_score.py`
- Contains: AMD hardware models, bound graphs, SOL bound artifacts, score reports, baseline artifacts, trace summaries.
- Depends on: Core data models, hardware model JSON in `src/sol_execbench/data/amd_hardware_models/`.
- Used by: `scripts/run_dataset.py`, tests in `tests/sol_execbench/test_amd_*.py`, reporting tests.

## Data Flow

### Primary Request Path

1. `sol-execbench` dispatches to `_evaluate_cli` from the package script configured in `pyproject.toml:27` and implemented in `src/sol_execbench/cli/main.py:233`.
2. The CLI loads `Definition`, `Workload`, `Solution`, and `BenchmarkConfig` using helpers in `src/sol_execbench/cli/main.py:53`, `src/sol_execbench/cli/main.py:57`, `src/sol_execbench/cli/main.py:66`, and `src/sol_execbench/cli/main.py:78`.
3. `ProblemPackager` writes validated JSON and source files into a temporary staging directory in `src/sol_execbench/driver/problem_packager.py:91`.
4. For native ROCm language categories, `ProblemPackager.compile()` writes `build_ext.py`, injects offload architecture flags, and the CLI runs `python build_ext.py` in `src/sol_execbench/driver/problem_packager.py:151` and `src/sol_execbench/cli/main.py:303`.
5. `src/sol_execbench/driver/templates/build_ext.py:1` loads `Solution`, compiles `.hip`/C++ sources, and produces `benchmark_kernel.so`.
6. `ProblemPackager.execute()` writes `eval_driver.py` and returns `python eval_driver.py` in `src/sol_execbench/driver/problem_packager.py:182`.
7. `eval_driver.py` loads staged JSON, imports core benchmark utilities, imports reference/user code, runs per-workload correctness and timing loops, and emits strict JSON traces with `_emit` in `src/sol_execbench/driver/templates/eval_driver.py:292`.
8. The CLI parses stdout JSONL back into `Trace` models using `convert_stdout_to_traces` in `src/sol_execbench/driver/problem_packager.py:207`, then prints, writes, or returns JSONL from `src/sol_execbench/cli/main.py:367`.

### Dataset Execution Flow

1. `scripts/run_dataset.py:95` discovers problem directories under category roots containing `definition.json` and `workload.jsonl`.
2. `scripts/run_dataset.py:159` selects a provided solution JSON/Python file or falls back to wrapping `Definition.reference` through `scripts/run_dataset.py:209`.
3. `scripts/run_dataset.py:246` builds a `sol-execbench --json` command, and `scripts/run_dataset.py:279` runs it as a subprocess.
4. `scripts/run_dataset.py:394` inspects returned trace dictionaries for pass/fail summaries.
5. `scripts/run_dataset.py:502` optionally builds AMD-native score reports from traces, workload rows, baseline artifacts, AMD SOL bound artifacts, and solar derivation evidence.
6. `scripts/run_dataset.py:917` writes execution-closure sidecars that link readiness data, traces, summaries, CLI logs, solution refs, and evidence refs.

### Dataset Mapping And Readiness Flow

1. `scripts/download_solexecbench.py:190` downloads or materializes dataset files into `data/`.
2. `src/sol_execbench/core/dataset/manifest.py:107` builds a deterministic manifest from `inspect_dataset_layout`.
3. `src/sol_execbench/core/dataset/inventory.py:236` parses `Definition` and `Workload` rows into per-problem inventory records.
4. `src/sol_execbench/core/dataset/readiness.py:176` classifies workload readiness using inventory details.
5. `src/sol_execbench/core/dataset/ready_subset.py:61` selects ready workload refs.
6. `src/sol_execbench/core/dataset/parity_gap.py:216` builds parity-gap reports from manifest, inventory, readiness, ready-subset, and closure artifacts.

**State Management:**
- Runtime state is mostly local to model instances, staging directories, subprocesses, and sidecar files.
- `Solution` is frozen and memoizes its content hash in `src/sol_execbench/core/data/solution.py:265`.
- Generated eval driver module globals in `src/sol_execbench/driver/templates/eval_driver.py` hold benchmark state for one subprocess only.
- Module-level constants define schemas, claim boundaries, language sets, warning messages, and rule tables across `src/sol_execbench/core/`, `src/sol_execbench/driver/`, and `scripts/run_dataset.py`.

## Key Abstractions

**Definition:**
- Purpose: Formal workload contract: symbolic axes, inputs, outputs, reference implementation, and optional custom input factory.
- Examples: `src/sol_execbench/core/data/definition.py`, `examples/pytorch/gemma3_swiglu/definition.json`, `tests/sol_execbench/samples/rmsnorm/definition.json`
- Pattern: Pydantic model with AST validators for reference code and ordered input contract.

**Workload:**
- Purpose: Concrete axis values, input generation descriptors, UUID, and numeric tolerance for one run.
- Examples: `src/sol_execbench/core/data/workload.py`, `examples/triton/rmsnorm/workload.jsonl`
- Pattern: Pydantic model with discriminated input-spec unions and tolerance defaults.

**Solution:**
- Purpose: Source bundle plus language, target hardware, entry point, binding, and compile options.
- Examples: `src/sol_execbench/core/data/solution.py`, `examples/hip_cpp/rmsnorm/solution_hip.json`, `examples/triton/rmsnorm/solution_triton.json`
- Pattern: Frozen Pydantic model with source-path security validation, ROCm-only language validation, and deterministic hash.

**Trace:**
- Purpose: Canonical benchmark output connecting definition, workload, optional solution name, and optional evaluation result.
- Examples: `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/reporting.py`
- Pattern: Pydantic model with status-dependent correctness/performance validation.

**BenchmarkConfig:**
- Purpose: Runtime knobs for warmup, iterations, seed, reference benchmarking, and clock-lock behavior.
- Examples: `src/sol_execbench/core/bench/config/benchmark_config.py`, `src/sol_execbench/core/bench/config/device_config.py`
- Pattern: Dataclass-style config imported through `src/sol_execbench/core/__init__.py`.

**ProblemPackager:**
- Purpose: Staging-boundary object that writes model JSON and sources, injects target flags, and returns subprocess commands.
- Examples: `src/sol_execbench/driver/problem_packager.py`
- Pattern: Stateful per-run packager with temporary directory ownership and generated driver templates.

**AMD SOL And Score Artifacts:**
- Purpose: Derived evidence for AMD-native performance claims using bound graphs, hardware models, baselines, and traces.
- Examples: `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/amd_score.py`
- Pattern: Frozen dataclasses with `to_dict()` serialization and explicit warnings/claim levels.

**Dataset Sidecars:**
- Purpose: Deterministic metadata and readiness artifacts stored outside canonical traces.
- Examples: `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/inventory.py`, `src/sol_execbench/core/dataset/readiness.py`, `src/sol_execbench/core/dataset/parity_gap.py`
- Pattern: Pydantic sidecar models with checksum fields and writer functions.

## Entry Points

**Evaluator CLI:**
- Location: `src/sol_execbench/cli/main.py`
- Triggers: `sol-execbench` console script from `pyproject.toml:27`
- Responsibilities: Evaluate one problem directory or explicit definition/workload/solution files and optionally print contract metadata.

**Baseline CLI:**
- Location: `src/sol_execbench/cli/baseline.py`
- Triggers: `sol-execbench-baseline` console script from `pyproject.toml:29`
- Responsibilities: Compare traces to baseline artifacts and surface claim guardrails.

**Dataset Runner:**
- Location: `scripts/run_dataset.py`
- Triggers: `uv run scripts/run_dataset.py ...`
- Responsibilities: Batch problems, construct solutions, invoke evaluator CLI, collect reports, scores, timing evidence, and execution closure data.

**Dataset Download:**
- Location: `scripts/download_solexecbench.py`
- Triggers: `uv run scripts/download_solexecbench.py ...`
- Responsibilities: Build local SOL ExecBench dataset files and manifest sidecars from upstream data.

**Dataset Inspection:**
- Location: `scripts/inspect_dataset.py`
- Triggers: `uv run scripts/inspect_dataset.py ...`
- Responsibilities: Build manifest, inventory, readiness, and ready-subset sidecars for a local dataset root.

**Parity Gap Report:**
- Location: `scripts/report_parity_gaps.py`
- Triggers: `uv run scripts/report_parity_gaps.py ...`
- Responsibilities: Generate parity-gap JSON and Markdown from dataset sidecars.

**Generated Evaluation Process:**
- Location: `src/sol_execbench/driver/templates/eval_driver.py`
- Triggers: `ProblemPackager.execute()` from `src/sol_execbench/driver/problem_packager.py:182`
- Responsibilities: Run reference and submitted code in the staging process and emit trace JSONL.

## Architectural Constraints

- **Threading:** Main CLI orchestration is synchronous subprocess execution. The generated eval driver records thread counts around timing and rejects thread injection via `check_thread_injection` in `src/sol_execbench/core/bench/reward_hack.py:249`.
- **GPU API naming:** Use PyTorch's `torch.cuda` APIs for ROCm device availability, events, synchronization, and extension build integration because PyTorch ROCm exposes HIP behavior through this namespace. Do not introduce a parallel `torch.hip` abstraction unless PyTorch provides one and the existing tests are updated.
- **Global state:** `src/sol_execbench/driver/templates/eval_driver.py` uses module-level state for staging paths, parsed models, imported functions, integrity snapshots, and per-process config. Keep this state confined to the generated subprocess.
- **Generated template imports:** `src/sol_execbench/driver/templates/eval_driver.py` redirects stdout before importing torch, then writes trace JSON to the duplicated original stdout. Preserve this split so library logs do not corrupt JSONL.
- **Canonical output:** `Trace` JSONL is the benchmark output. Additional reports must live in sidecars and reference traces instead of adding fields to `Trace`.
- **ROCm-only schema:** `BuildSpec` in `src/sol_execbench/core/data/solution.py:132` rejects legacy CUDA/NVIDIA language and compile-option values. New solution categories must be added there and in native language sets in `src/sol_execbench/driver/problem_packager.py` and `src/sol_execbench/driver/templates/eval_driver.py`.
- **Dataset sidecars:** Sidecar writers should use deterministic JSON, stable checksums, and explicit claim boundaries, following `src/sol_execbench/core/dataset/manifest.py` and related modules.
- **Circular imports:** Not detected in the inspected layer graph. Keep `src/sol_execbench/core/data/` independent of driver and CLI modules.

## Anti-Patterns

### Running User Code In The CLI Process

**What happens:** User solution or reference code is imported directly from `src/sol_execbench/cli/main.py`.
**Why it's wrong:** It bypasses stdout isolation, reward-hack checks, staging cleanup, and compile/evaluation timeout boundaries.
**Do this instead:** Stage files through `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py` and execute `eval_driver.py` as a subprocess.

### Extending Trace For Derived Evidence

**What happens:** Score, readiness, timing evidence, or parity status fields are added to `Trace`.
**Why it's wrong:** `Trace` is the canonical benchmark output schema in `src/sol_execbench/core/data/trace.py`; derived data has different claim boundaries and evidence references.
**Do this instead:** Use sidecar models in `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/scoring/`, or `src/sol_execbench/core/dataset/`.

### Duplicating Schema Parsing In Scripts

**What happens:** Scripts manually inspect JSON dicts without validating `Definition`, `Workload`, or `Solution` models.
**Why it's wrong:** This skips validators for reference entry points, shape axes, source paths, language compatibility, and status invariants.
**Do this instead:** Instantiate models from `src/sol_execbench/core/data/` before using schema data.

### Adding New ROCm Languages In Only One Place

**What happens:** A new language value is added to examples or docs but not to `SupportedLanguages`, `ProblemPackager`, and the generated eval driver.
**Why it's wrong:** Native compilation and import dispatch depend on parallel native-language sets.
**Do this instead:** Update `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/eval_driver.py`, examples, and tests together.

## Error Handling

**Strategy:** Validate early with typed models, convert predictable user-facing CLI failures into Click exceptions or trace statuses, and isolate runtime failures into per-workload `EvaluationStatus` values when possible.

**Patterns:**
- CLI argument and missing-file failures raise `click.ClickException` in `src/sol_execbench/cli/main.py`.
- Pydantic validators raise `ValueError` for schema contract violations in `src/sol_execbench/core/data/`.
- Evaluation failures become `Trace` objects with statuses such as `INVALID_REFERENCE`, `RUNTIME_ERROR`, `INCORRECT_SHAPE`, `INCORRECT_DTYPE`, `INCORRECT_NUMERICAL`, and `REWARD_HACK` from `src/sol_execbench/core/data/trace.py`.
- Generated eval driver catches reference, input generation, user function, timing, and reward-hack failures per workload in `src/sol_execbench/driver/templates/eval_driver.py`.
- Dataset scripts save subprocess stdout/stderr logs when no traces are parsed in `scripts/run_dataset.py:321`.
- Scoring functions preserve incomplete evidence as `score=None` plus warnings instead of manufacturing scores in `src/sol_execbench/core/scoring/amd_score.py`.

## Cross-Cutting Concerns

**Logging:** CLI human output uses Rich in `src/sol_execbench/cli/main.py`; generated eval driver redirects third-party stdout to stderr and writes only JSON traces to original stdout; dataset scripts use print/log sidecars in `scripts/run_dataset.py`.
**Validation:** Pydantic schemas in `src/sol_execbench/core/data/` and `src/sol_execbench/core/dataset/` define contracts; tests assert public guardrails in `tests/sol_execbench/test_public_contract_guardrails.py` and related files.
**Authentication:** Not applicable in package runtime. Dataset download may access Hugging Face through `datasets` in `scripts/download_solexecbench.py`, but no in-repo auth provider exists.
**Security:** Source path traversal is blocked by `SourceFile` in `src/sol_execbench/core/data/solution.py`; generated eval driver blocks Python-side dynamic native loading and static reward-hack patterns through `src/sol_execbench/core/bench/reward_hack.py`.
**Performance:** Timing uses PyTorch ROCm device events and a shifting memory pool allocator in `src/sol_execbench/core/bench/timing.py` and `src/sol_execbench/core/bench/io.py`.
**Hardware Claims:** AMD score and dataset readiness code use explicit claim levels, warnings, hardware-model refs, and evidence refs in `src/sol_execbench/core/scoring/` and `src/sol_execbench/core/dataset/`.

---

*Architecture analysis: 2026-05-24*
