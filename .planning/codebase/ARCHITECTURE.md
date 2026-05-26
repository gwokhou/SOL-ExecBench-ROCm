<!-- refreshed: 2026-05-26 -->
# Architecture

**Analysis Date:** 2026-05-26

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
├────────────────────┬────────────────────┬───────────────────┤
│ Evaluator command  │ Baseline command   │ Batch scripts     │
│ `src/sol_execbench/│ `src/sol_execbench/│ `scripts/`        │
│ cli/main.py`       │ cli/baseline.py`   │                   │
└──────────┬─────────┴─────────┬──────────┴─────────┬─────────┘
           │                   │                    │
           ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                       Core Package                          │
│ `src/sol_execbench/core/`                                   │
├────────────────────┬────────────────────┬───────────────────┤
│ Data contracts     │ Bench runtime      │ Dataset/scoring   │
│ `core/data/`       │ `core/bench/`      │ `core/dataset/`   │
│                    │                    │ `core/scoring/`   │
└──────────┬─────────┴─────────┬──────────┴─────────┬─────────┘
           │                   │                    │
           ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Driver/Staging Layer                      │
│ `src/sol_execbench/driver/problem_packager.py`              │
│ `src/sol_execbench/driver/templates/`                       │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Runtime Outputs and Assets                                  │
│ temp staging dirs, trace JSONL, sidecars, `data/`, `out/`   │
└─────────────────────────────────────────────────────────────┘
```

SOL ExecBench ROCm is a layered Python package. The CLI reads benchmark artifacts, validates them through Pydantic models, packages each run into a temporary staging directory, optionally compiles native HIP/C++ code, runs a generated evaluation driver in a subprocess, parses trace JSONL, and writes optional profiling/static-evidence/environment sidecars. Dataset and scoring utilities consume the same contracts for larger benchmark runs and release evidence.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Evaluator CLI | Resolves input paths, loads contracts, creates staging dirs, runs compile/evaluate subprocesses, writes trace and sidecar outputs | `src/sol_execbench/cli/main.py` |
| Metadata subcommands | Dispatches `contract`, `doctor`, and `toolchain` subcommands from the same console script | `src/sol_execbench/cli/main.py` |
| Baseline CLI | Compares trace JSONL against baseline trace JSONL and renders JSON/text summaries | `src/sol_execbench/cli/baseline.py` |
| Public package facade | Re-exports contract, environment, toolchain, and bench types for callers and templates | `src/sol_execbench/core/__init__.py` |
| Data contracts | Defines benchmark definitions, workloads, solutions, traces, evaluator contract metadata, dtypes, and shape resolution | `src/sol_execbench/core/data/` |
| Bench runtime helpers | Generates inputs, checks correctness, times GPU callables, enforces clock policy, detects reward hacks, and collects optional ROCm evidence | `src/sol_execbench/core/bench/` |
| Driver packager | Writes normalized inputs and solution sources into staging, injects HIP offload arch flags, and returns compile/evaluate commands | `src/sol_execbench/driver/problem_packager.py` |
| Generated build driver | Builds native ROCm extension artifacts with `torch.utils.cpp_extension.load()` | `src/sol_execbench/driver/templates/build_ext.py` |
| Generated eval driver | Imports references and submitted code, runs correctness/timing checks per workload, emits strict trace JSONL | `src/sol_execbench/driver/templates/eval_driver.py` |
| Dataset helpers | Inspects benchmark roots, builds manifests, classifies ROCm readiness, creates ready subsets, and reports parity gaps | `src/sol_execbench/core/dataset/` |
| Scoring helpers | Builds AMD bound graphs, estimates work, derives SOL/SOLAR evidence, and computes guarded AMD-native scores | `src/sol_execbench/core/scoring/` |
| Batch runner | Discovers problem directories, builds solution JSON, invokes the CLI repeatedly, and writes closure/scoring sidecars | `scripts/run_dataset.py` |

## Pattern Overview

**Overall:** Layered CLI package with process-isolated evaluation drivers and typed JSON contracts.

**Key Characteristics:**
- Use Pydantic models in `src/sol_execbench/core/data/` as the boundary between files, CLI options, generated drivers, traces, dataset reports, and scoring evidence.
- Keep submitted solution execution out of the CLI process by staging files with `ProblemPackager` and running `build_ext.py` / `eval_driver.py` via `subprocess.run()`.
- Keep ROCm-specific hardware, timing, profiling, and toolchain concerns in `src/sol_execbench/core/bench/`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, and `src/sol_execbench/core/diagnostics.py`.
- Treat `scripts/` as orchestration around the package API and console scripts, not as a replacement for reusable core logic.

## Layers

**CLI Layer:**
- Purpose: Provide user-facing commands and translate CLI arguments/files into typed package objects.
- Location: `src/sol_execbench/cli/`
- Contains: Click commands, Rich output, subprocess orchestration, output/sidecar writing, baseline comparison command.
- Depends on: `src/sol_execbench/core/`, `src/sol_execbench/driver/`, standard library subprocess/tempfile/path utilities.
- Used by: Console scripts in `pyproject.toml`, tests in `tests/sol_execbench/`, batch runners in `scripts/`.

**Contract Layer:**
- Purpose: Define stable schema objects for benchmark input, solution metadata, workload rows, trace output, and compatibility metadata.
- Location: `src/sol_execbench/core/data/`
- Contains: `Definition`, `Workload`, `Solution`, `Trace`, enums, validators, dtype and shape helpers.
- Depends on: Pydantic, Python AST parsing, small dtype/shape helpers.
- Used by: CLI loading, driver templates, dataset inventory/readiness, scoring, docs/tests.

**Bench Runtime Layer:**
- Purpose: Execute one workload correctly and reproducibly once inside the generated evaluation subprocess.
- Location: `src/sol_execbench/core/bench/`
- Contains: input generation, safetensors loading, output normalization/allocation, correctness metrics, timing, clock locking, timing policy, reward-hack defense, ROCm profiler/static evidence helpers.
- Depends on: PyTorch ROCm, package data contracts, ROCm tools when optional evidence is requested.
- Used by: `src/sol_execbench/driver/templates/eval_driver.py` and CLI evidence paths.

**Driver/Staging Layer:**
- Purpose: Convert typed problem objects into an isolated runnable directory and generated command list.
- Location: `src/sol_execbench/driver/`
- Contains: `ProblemPackager`, `build_ext.py`, `eval_driver.py`.
- Depends on: core contracts, solution language enums, ROCm local-gfx discovery tooling, PyTorch extension build API in the generated build template.
- Used by: `src/sol_execbench/cli/main.py`.

**Dataset Layer:**
- Purpose: Inspect downloaded benchmark assets and produce auditable dataset sidecars.
- Location: `src/sol_execbench/core/dataset/`
- Contains: layout inspection, checksums, manifest models, inventory records, readiness classification, ready subset generation, parity gap reports.
- Depends on: core contracts and local filesystem paths.
- Used by: `scripts/inspect_dataset.py`, `scripts/download_solexecbench.py`, `scripts/report_parity_gaps.py`, `scripts/run_dataset.py`, tests.

**Scoring Layer:**
- Purpose: Derive AMD-native performance evidence and guarded score artifacts from definitions, workloads, traces, hardware models, and baselines.
- Location: `src/sol_execbench/core/scoring/`
- Contains: bound graphs, work estimates, hardware models, AMD SOL v1/v2, SOLAR derivation evidence, baseline artifacts, AMD score reports.
- Depends on: core contracts, packaged hardware model JSON in `src/sol_execbench/data/amd_hardware_models/`, reporting helpers.
- Used by: `scripts/run_dataset.py`, tests, reporting docs.

**Operational Assets:**
- Purpose: Provide examples, tests, container support, docs, and downloaded datasets around the Python package.
- Location: `examples/`, `tests/`, `docker/`, `docs/`, `data/`
- Contains: runnable problem fixtures, pytest coverage, ROCm Docker environment, user/research documentation, local benchmark assets.
- Depends on: package code and external ROCm/PyTorch environment.
- Used by: contributors, CI, GPU evaluation workflows.

## Data Flow

### Primary Request Path

1. CLI receives either `PROBLEM_DIR` or explicit `--definition`, `--workload`, and `--solution` inputs (`src/sol_execbench/cli/main.py:502`).
2. Problem directory paths are resolved to `definition.json`, `workload.jsonl`, optional `config.json`, and optional `solution.json` (`src/sol_execbench/cli/main.py:596`).
3. Files are loaded into `Definition`, `Workload`, `Solution`, and `BenchmarkConfig` models (`src/sol_execbench/cli/main.py:613`).
4. A temporary staging directory is created and handed to `ProblemPackager` (`src/sol_execbench/cli/main.py:630`).
5. `ProblemPackager.__init__()` writes normalized `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, and source files into the staging directory (`src/sol_execbench/driver/problem_packager.py:91`).
6. Native ROCm categories compile first through generated `build_ext.py`; offload arch flags may be injected from `target_hardware` or local ROCm tooling (`src/sol_execbench/driver/problem_packager.py:131`, `src/sol_execbench/driver/problem_packager.py:175`).
7. The CLI stages and runs `eval_driver.py` in a subprocess, optionally wrapped by `rocprofv3` evidence collection (`src/sol_execbench/cli/main.py:692`).
8. The generated eval driver loads staged JSON files, validates source review, imports reference/user code, runs correctness rounds, measures latency, and emits trace JSONL (`src/sol_execbench/driver/templates/eval_driver.py:89`, `src/sol_execbench/driver/templates/eval_driver.py:176`).
9. CLI parses stdout JSON lines into `Trace` objects and writes optional trace/sidecar outputs (`src/sol_execbench/cli/main.py:737`).
10. CLI exits with status `0` only when all traces have `EvaluationStatus.PASSED` (`src/sol_execbench/cli/main.py:764`).

### Native ROCm Compile Flow

1. `Solution.spec.languages` determines whether a solution is native ROCm (`hip_cpp`, `hipblas`, `miopen`, `ck`, or `rocwmma`) (`src/sol_execbench/driver/problem_packager.py:43`).
2. `ProblemPackager.compile()` rewrites `solution.json` with injected HIP flags and copies `build_ext.py` into staging (`src/sol_execbench/driver/problem_packager.py:175`).
3. `build_ext.py` validates the solution contract, gathers `.hip`, `.cpp`, `.cc`, `.cxx`, and `.c` source files in staging, sets `PYTORCH_ROCM_ARCH` from target hardware when available, calls `torch.utils.cpp_extension.load()`, and normalizes the output to `benchmark_kernel.so` (`src/sol_execbench/driver/templates/build_ext.py`).
4. `ProblemPackager.execute()` refuses to stage evaluation for native ROCm code unless `benchmark_kernel.so` exists (`src/sol_execbench/driver/problem_packager.py:203`).
5. The eval driver imports the compiled extension as `benchmark_kernel` and resolves the configured entry function (`src/sol_execbench/driver/templates/eval_driver.py:220`).

### Dataset Batch Flow

1. `scripts/run_dataset.py` discovers problem directories or category roots, optionally restricts categories, limits problems/workloads, and builds reference or custom solution JSON (`scripts/run_dataset.py:95`, `scripts/run_dataset.py:159`).
2. The runner builds a `sol-execbench` command for each problem and captures trace JSONL and logs (`scripts/run_dataset.py:246`, `scripts/run_dataset.py:279`).
3. It can collect static/timing evidence, derive AMD score reports, SOL bounds, SOLAR derivations, and execution closure records (`scripts/run_dataset.py:333`, `scripts/run_dataset.py:502`, `scripts/run_dataset.py:917`).
4. Dataset sidecar builders in `src/sol_execbench/core/dataset/` provide layout, manifest, inventory, readiness, ready-subset, and parity-gap models consumed by scripts and tests.

**State Management:**
- Benchmark state is file-backed: JSON/JSONL inputs, staged source files, generated `.so` artifacts, trace JSONL, and sidecar JSON files.
- Runtime state for one evaluation lives in the generated subprocess, not in the parent CLI process.
- Module-level constants are used for command names, environment variable names, supported language sets, timing policy values, and schema versions; avoid adding mutable global runtime state.
- `ProblemPackager.__del__()` removes staging directories unless `keep_output_dir=True` (`src/sol_execbench/driver/problem_packager.py:123`).

## Key Abstractions

**Definition:**
- Purpose: Machine-readable problem contract with symbolic axes, input/output tensor specs, and executable Python reference code.
- Examples: `src/sol_execbench/core/data/definition.py`, `examples/pytorch/gemma3_swiglu/definition.json`
- Pattern: Pydantic model with AST validators requiring a top-level `run()` reference and matching input names (`src/sol_execbench/core/data/definition.py:136`).

**Workload:**
- Purpose: Concrete axis values, input generation descriptors, UUID, and tolerance for one benchmark row.
- Examples: `src/sol_execbench/core/data/workload.py`, `examples/triton/rmsnorm/workload.jsonl`
- Pattern: Pydantic model with discriminated input descriptor union for random, scalar, safetensors, and custom inputs (`src/sol_execbench/core/data/workload.py:27`, `src/sol_execbench/core/data/workload.py:102`).

**Solution and BuildSpec:**
- Purpose: Submitted implementation metadata, source files, languages, target hardware, entry point, compile options, and destination-passing style.
- Examples: `src/sol_execbench/core/data/solution.py`, `examples/hip_cpp/rmsnorm/solution_hip.json`
- Pattern: Pydantic model and enums that reject legacy CUDA/NVIDIA schema values and enforce Python-vs-native language separation (`src/sol_execbench/core/data/solution.py:29`, `src/sol_execbench/core/data/solution.py:132`).

**Trace:**
- Purpose: Per-workload output record linking definition, workload, solution name, evaluation status, correctness, performance, environment, timestamp, and logs.
- Examples: `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/baseline.py`
- Pattern: Pydantic model with status-dependent validation of correctness/performance fields (`src/sol_execbench/core/data/trace.py:86`, `src/sol_execbench/core/data/trace.py:113`, `src/sol_execbench/core/data/trace.py:176`).

**ProblemPackager:**
- Purpose: File-staging boundary between trusted CLI/model loading and untrusted submitted code execution.
- Examples: `src/sol_execbench/driver/problem_packager.py`
- Pattern: Constructor writes normalized state up front, `compile()` and `execute()` stage generated templates and return argv lists for parent subprocess execution.

**Bench Helpers:**
- Purpose: Runtime primitives for generated driver execution.
- Examples: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/reward_hack.py`
- Pattern: Function-oriented helpers imported by `eval_driver.py` so the generated script stays mostly orchestration code.

**Dataset Evidence Models:**
- Purpose: Structured audit artifacts for downloaded benchmark coverage, ROCm readiness, and parity gaps.
- Examples: `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/inventory.py`, `src/sol_execbench/core/dataset/readiness.py`, `src/sol_execbench/core/dataset/parity_gap.py`
- Pattern: Pydantic sidecar models plus writer functions that serialize deterministic JSON.

**AMD Score Artifacts:**
- Purpose: Translate traces and static definition/workload evidence into guarded AMD-native scoring outputs.
- Examples: `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_score.py`
- Pattern: Build bound graph, estimate operator work, combine hardware model and baseline evidence, then report status/guardrails.

## Entry Points

**Evaluator CLI:**
- Location: `src/sol_execbench/cli/main.py`
- Triggers: `sol-execbench` console script from `pyproject.toml`.
- Responsibilities: Evaluate one problem, print tables or JSON, write trace JSONL, collect optional environment/profile/static sidecars.

**CLI Metadata Subcommands:**
- Location: `src/sol_execbench/cli/main.py`
- Triggers: `sol-execbench contract --json`, `sol-execbench doctor --json`, `sol-execbench toolchain --json`.
- Responsibilities: Print GPU-free evaluator contract metadata, ROCm environment diagnostics, and toolchain routing decisions.

**Baseline CLI:**
- Location: `src/sol_execbench/cli/baseline.py`
- Triggers: `sol-execbench-baseline` console script from `pyproject.toml`.
- Responsibilities: Load current/baseline trace JSONL and classify regressions/improvements.

**Batch Dataset Runner:**
- Location: `scripts/run_dataset.py`
- Triggers: `uv run scripts/run_dataset.py <problems_dir>`.
- Responsibilities: Discover benchmark problems, build solution inputs, invoke `sol-execbench`, collect traces/logs, and optionally write AMD scoring/closure sidecars.

**Dataset Inspector:**
- Location: `scripts/inspect_dataset.py`
- Triggers: `uv run scripts/inspect_dataset.py <dataset_root>`.
- Responsibilities: Build layout/inventory/readiness outputs from benchmark asset roots.

**Generated Evaluation Driver:**
- Location: `src/sol_execbench/driver/templates/eval_driver.py`
- Triggers: Copied to staging and run as `python eval_driver.py` by the CLI.
- Responsibilities: Execute the reference and user solution for each workload, enforce guardrails, measure latency, and emit strict JSONL traces.

## Architectural Constraints

- **Threading:** Parent CLI is synchronous. Generated evaluation checks thread count to detect reward-hack thread injection around timing (`src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/reward_hack.py`).
- **Global state:** `eval_driver.py` mutates process stdout before importing torch, inserts the staging directory into `sys.path`, monkey-patches `torch.utils.cpp_extension.load/load_inline` for Python submissions, and keeps timing-integrity snapshots in module globals. Keep these mutations inside the generated subprocess only.
- **Circular imports:** Core package facades (`src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/data/__init__.py`, `src/sol_execbench/core/scoring/__init__.py`) re-export many symbols. New modules should import concrete implementation modules when facade imports risk cycles.
- **Device naming:** PyTorch ROCm still exposes HIP devices through `torch.cuda`; architecture docs and code should treat `torch.cuda` calls in timing/runtime code as ROCm-compatible rather than CUDA backend support (`src/sol_execbench/core/bench/timing.py`).
- **Staging cleanup:** Staging directories are temporary by default and removed by `ProblemPackager.__del__()`; debugging workflows requiring artifacts must pass `--keep-staging`.
- **Secrets:** Environment files and downloaded datasets are operational inputs only. Do not read or embed `.env*` contents or proprietary benchmark data in docs/tests.

## Anti-Patterns

### Importing Submitted Code in the Parent CLI

**What happens:** A new CLI feature imports or executes solution source directly from `src/sol_execbench/cli/main.py`.
**Why it's wrong:** It bypasses staging isolation, stdout separation, reward-hack checks, and subprocess failure containment.
**Do this instead:** Stage source through `ProblemPackager` and add runtime behavior to `src/sol_execbench/driver/templates/eval_driver.py` or `src/sol_execbench/core/bench/`.

### Adding Benchmark Logic to Scripts Only

**What happens:** Dataset or scoring behavior is implemented only in `scripts/run_dataset.py`.
**Why it's wrong:** Tests, CLI paths, and docs cannot reuse it, and schema behavior drifts from core models.
**Do this instead:** Put reusable logic in `src/sol_execbench/core/dataset/`, `src/sol_execbench/core/scoring/`, or `src/sol_execbench/core/reporting.py`, then call it from scripts.

### Mixing Python and Native ROCm Solution Categories

**What happens:** A solution spec attempts to list both `pytorch`/`triton` and `hip_cpp`/`hipblas`/`miopen`/`ck`/`rocwmma`.
**Why it's wrong:** The driver uses separate Python import and native extension execution paths, and `BuildSpec` rejects mixed categories.
**Do this instead:** Use a single language family and place native compilation metadata in `Solution.spec.compile_options` (`src/sol_execbench/core/data/solution.py:210`).

### Bypassing Typed Contracts

**What happens:** New code parses benchmark JSON into dictionaries and passes them across layers.
**Why it's wrong:** Contract validation, ROCm migration guardrails, trace invariants, and output serialization can be skipped.
**Do this instead:** Instantiate `Definition`, `Workload`, `Solution`, `BenchmarkConfig`, and `Trace` models at boundaries, following `src/sol_execbench/cli/main.py:613`.

## Error Handling

**Strategy:** Convert known benchmark outcomes into `Trace.evaluation.status` values inside the evaluation subprocess; reserve parent CLI non-zero exits for infrastructure failures, missing traces, compile failure, or any non-passing workload result.

**Patterns:**
- Use `click.ClickException` for invalid CLI arguments or missing input paths (`src/sol_execbench/cli/main.py`).
- Use Pydantic validation errors to reject invalid benchmark schemas before staging (`src/sol_execbench/core/data/`).
- Use `EvaluationStatus` for workload-level outcomes such as `INVALID_REFERENCE`, `INCORRECT_SHAPE`, `INCORRECT_DTYPE`, `INCORRECT_NUMERICAL`, `RUNTIME_ERROR`, `COMPILE_ERROR`, `TIMEOUT`, and `REWARD_HACK` (`src/sol_execbench/core/data/trace.py:86`).
- Optional diagnostics/evidence collection should report warnings or sidecar statuses without changing correctness unless the benchmark contract explicitly requires it.
- Strict JSON trace emission uses `allow_nan=False` in the generated driver so non-standard NaN/Inf trace payloads fail immediately (`src/sol_execbench/driver/templates/eval_driver.py`).

## Cross-Cutting Concerns

**Logging:** CLI user output uses Rich `Console` in `src/sol_execbench/cli/main.py`; generated evaluation redirects non-JSON stdout to stderr before importing torch/triton so parent stdout remains parseable JSONL.

**Validation:** Pydantic models validate input contracts in `src/sol_execbench/core/data/`; solution schema validators reject CUDA/NVIDIA migration residue and unsafe source paths.

**Authentication:** Not applicable; the package runs local CLI workflows and does not implement user identity or network-authenticated services.

**Security:** Submitted source paths cannot be absolute or contain `..`; dynamic native loading is blocked for Python submissions inside `eval_driver.py`; static source review blocks file I/O, subprocess/network access, stream tricks, semantic output caching, and unauthorized native loading patterns.

**ROCm Evidence:** Environment, toolchain, static-kernel, timing-policy, and rocprofv3 helpers live under `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/bench/timing_policy.py`, and `src/sol_execbench/core/bench/rocm_profiler.py`.

---

*Architecture analysis: 2026-05-26*
