<!-- refreshed: 2026-05-31 -->
# Architecture

**Analysis Date:** 2026-05-31

## System Overview

```text
+-------------------------------------------------------------+
|                       Command Layer                         |
|  Evaluator CLI: `src/sol_execbench/cli/main.py`             |
|  Baseline CLI:  `src/sol_execbench/cli/baseline.py`         |
|  Dataset/report scripts: `scripts/`                         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                         Core Domain                         |
|  Schemas, benchmark helpers, dataset reports, scoring,       |
|  environment diagnostics, and toolchain routing              |
|  `src/sol_execbench/core/`                                   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                    Staging / Isolation Layer                |
|  `ProblemPackager` writes a temp problem bundle and          |
|  templates from `src/sol_execbench/driver/`                  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  GPU Evaluation Subprocess                  |
|  `eval_driver.py` imports user code, checks correctness,     |
|  measures timing, and emits strict JSONL `Trace` objects     |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  Outputs, Sidecars, Evidence                |
|  Trace JSONL, environment snapshots, rocprofv3 profiles,     |
|  static kernel evidence, and AMD-native score reports        |
+-------------------------------------------------------------+
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Root evaluator CLI | Dispatch `sol-execbench`, parse inputs, stage evaluation, compile HIP/C++ submissions, run profiling/static evidence, write traces and sidecars | `src/sol_execbench/cli/main.py` |
| Baseline CLI | Compare candidate trace JSONL against one or more baseline trace files | `src/sol_execbench/cli/baseline.py` |
| Public package exports | Re-export stable schema and utility APIs for library consumers | `src/sol_execbench/__init__.py`, `src/sol_execbench/core/__init__.py` |
| Data schemas | Define `Definition`, `Workload`, `Solution`, `Trace`, evaluator contract, dtype conversion, JSON helpers | `src/sol_execbench/core/data/` |
| Benchmark helpers | Generate inputs, allocate outputs, compute correctness, measure timing, detect reward hacks, collect ROCm profiler/static evidence | `src/sol_execbench/core/bench/` |
| Dataset reporting | Inspect downloaded benchmark layouts, inventories, readiness, parity gaps, execution closure, checksums | `src/sol_execbench/core/dataset/` |
| Scoring | Build guarded AMD SOL bounds and derived AMD-native score reports from traces and baseline artifacts | `src/sol_execbench/core/scoring/` |
| Environment diagnostics | Collect optional ROCm/PyTorch/tool snapshots and doctor diagnostics without changing canonical traces | `src/sol_execbench/core/environment.py` |
| Toolchain routing | Model ROCm tool capabilities and choose diagnostic routes for profiling/static evidence | `src/sol_execbench/core/toolchain.py` |
| Problem packager | Write solution sources and problem JSON into a staging directory, inject HIP offload architecture flags, return subprocess commands | `src/sol_execbench/driver/problem_packager.py` |
| Evaluation template | Self-contained GPU-side execution script copied into staging directories | `src/sol_execbench/driver/templates/eval_driver.py` |
| Build template | Torch C++ extension build script copied into staging directories for native ROCm submissions | `src/sol_execbench/driver/templates/build_ext.py` |
| Dataset runner | Batch-run single problems or dataset categories through the installed CLI and build score/evidence/closure reports | `scripts/run_dataset.py` |

## Pattern Overview

**Overall:** Layered CLI-driven benchmark runner with schema-first domain models and staged subprocess isolation.

**Key Characteristics:**
- Use Pydantic models in `src/sol_execbench/core/data/` as the public contract for problem definitions, workloads, submitted solutions, traces, and evaluator metadata.
- Keep the command layer in `src/sol_execbench/cli/` thin enough to orchestrate I/O, subprocesses, and optional sidecars while delegating domain checks to `src/sol_execbench/core/`.
- Run user code only from a generated staging directory through `src/sol_execbench/driver/templates/eval_driver.py`; do not import submitted solution code in the parent CLI process.
- Treat optional evidence as sidecars. Environment snapshots, `rocprofv3` profiles, static kernel evidence, and AMD-native derived scores are diagnostic or derived artifacts, not replacements for canonical `Trace` JSONL.
- Preserve PyTorch's historical `torch.cuda` API names as the ROCm execution surface where PyTorch ROCm exposes HIP-backed devices and timing events.

## Layers

**Command Layer:**
- Purpose: Parse user intent, validate file presence, invoke core APIs, run subprocesses, and render output.
- Location: `src/sol_execbench/cli/`, `scripts/`
- Contains: Click commands, argparse scripts, Rich tables, batch orchestration, report entry points.
- Depends on: `src/sol_execbench/core/`, `src/sol_execbench/driver/`, Python `subprocess`, filesystem paths.
- Used by: `pyproject.toml` console scripts, tests under `tests/sol_execbench/`, user workflows in `docs/`.

**Domain Schema Layer:**
- Purpose: Define and validate benchmark contracts before execution.
- Location: `src/sol_execbench/core/data/`
- Contains: `Definition`, axis specs, tensor specs, `Workload`, input specs, `Solution`, build specs, `Trace`, evaluator contract, JSON helpers.
- Depends on: Pydantic, standard Python typing, shape/dtype helpers.
- Used by: CLI loaders in `src/sol_execbench/cli/main.py`, staging in `src/sol_execbench/driver/problem_packager.py`, evaluation template, dataset scripts, tests.

**Benchmark Execution Helper Layer:**
- Purpose: Provide GPU execution primitives used inside the evaluation subprocess.
- Location: `src/sol_execbench/core/bench/`
- Contains: input generation (`io.py`), correctness (`correctness.py`), timing (`timing.py`), reward-hack defense (`reward_hack.py`), clock locking (`clock_lock.py`), profiler/static evidence helpers.
- Depends on: PyTorch ROCm, safetensors, subprocess-based ROCm tools, `src/sol_execbench/core/data/`.
- Used by: `src/sol_execbench/driver/templates/eval_driver.py`, CLI sidecar collection, dataset runner.

**Driver / Isolation Layer:**
- Purpose: Materialize an immutable problem bundle in a temp directory and execute generated scripts there.
- Location: `src/sol_execbench/driver/`
- Contains: `ProblemPackager`, `build_ext.py` template, `eval_driver.py` template.
- Depends on: `src/sol_execbench/core/`, JSON serialization, temporary directories, ROCm architecture probes.
- Used by: `src/sol_execbench/cli/main.py`.

**Dataset and Reporting Layer:**
- Purpose: Inspect downloaded SOL ExecBench datasets and produce structured release/evidence reports.
- Location: `src/sol_execbench/core/dataset/`, `scripts/`
- Contains: layout inspection, manifest checksums, inventory/readiness, ready subsets, parity gaps, paper denominator reports, execution closure reports.
- Depends on: Pydantic, JSON, stable checksum helpers, data schemas.
- Used by: `scripts/inspect_dataset.py`, `scripts/run_dataset.py`, report scripts under `scripts/`, tests under `tests/sol_execbench/`.

**Scoring Layer:**
- Purpose: Compute AMD-specific bound artifacts and derived score reports with explicit claim guardrails.
- Location: `src/sol_execbench/core/scoring/`, `src/sol_execbench/sol_score.py`
- Contains: AMD hardware models, bound graph, bound estimates, SOL v1/v2 artifacts, baseline artifacts, derived AMD-native scoring, solar derivation evidence.
- Depends on: `Trace`, `Definition`, `Workload`, hardware model JSON in `src/sol_execbench/data/amd_hardware_models/`.
- Used by: `scripts/run_dataset.py`, scoring/report scripts, tests under `tests/sol_execbench/`.

## Data Flow

### Primary Request Path

1. User invokes `sol-execbench` via `pyproject.toml` entry point `sol_execbench.cli:cli` (`pyproject.toml:28`).
2. `SolExecbenchCli.main` routes root evaluator calls or metadata subcommands (`src/sol_execbench/cli/main.py:848`).
3. `_evaluate_cli` resolves `problem_dir` or explicit `--definition`, `--workload`, `--solution`, and `--config` paths (`src/sol_execbench/cli/main.py:568`).
4. `_load_definition`, `_load_workloads`, `_load_solution`, and `_load_config` instantiate Pydantic/domain objects (`src/sol_execbench/cli/main.py:90`, `src/sol_execbench/cli/main.py:94`, `src/sol_execbench/cli/main.py:103`, `src/sol_execbench/cli/main.py:115`).
5. `ProblemPackager` writes `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, and sources into a temp staging directory (`src/sol_execbench/driver/problem_packager.py:91`).
6. For HIP/C++ categories, `ProblemPackager.compile()` injects target `--offload-arch` flags when needed and stages `build_ext.py`; the parent CLI runs it as a subprocess (`src/sol_execbench/driver/problem_packager.py:165`, `src/sol_execbench/cli/main.py:640`).
7. `ProblemPackager.execute()` stages `eval_driver.py` and returns `["python", "eval_driver.py"]` (`src/sol_execbench/driver/problem_packager.py:190`).
8. `_run_evaluation_command` runs the evaluation subprocess with `PYTORCH_ALLOC_CONF=expandable_segments:True` (`src/sol_execbench/cli/main.py:444`).
9. The evaluation driver loads staged problem files, imports reference/user code, reviews sources, performs correctness checks, times the user function, and emits one strict JSON `Trace` per workload (`src/sol_execbench/driver/templates/eval_driver.py:90`, `src/sol_execbench/driver/templates/eval_driver.py:297`).
10. `ProblemPackager.convert_stdout_to_traces` parses JSONL stdout into `Trace` models (`src/sol_execbench/driver/problem_packager.py:212`).
11. The CLI writes trace JSONL, optional environment/profile/static evidence sidecars, and exits nonzero if any trace status is not `PASSED` (`src/sol_execbench/cli/main.py:744`).

### Metadata and Diagnostics Flow

1. `sol-execbench contract --json` calls `build_evaluator_contract()` and prints GPU-free compatibility metadata (`src/sol_execbench/cli/main.py:773`, `src/sol_execbench/core/data/contract.py:67`).
2. `sol-execbench doctor --json` calls `build_environment_diagnostics()` to collect bounded ROCm/PyTorch probes (`src/sol_execbench/cli/main.py:784`, `src/sol_execbench/core/environment.py:249`).
3. `sol-execbench toolchain --json` calls `build_toolchain_routing_report()` or lists `default_toolchain_registry()` (`src/sol_execbench/cli/main.py:817`, `src/sol_execbench/core/toolchain.py:204`, `src/sol_execbench/core/toolchain.py:376`).

### Dataset Batch Flow

1. `scripts/run_dataset.py` discovers problem directories under a single problem or category dataset root (`scripts/run_dataset.py:96`).
2. It builds a reference or custom `Solution` JSON wrapper when a solution file is absent or Python source is supplied (`scripts/run_dataset.py:160`, `scripts/run_dataset.py:182`, `scripts/run_dataset.py:210`).
3. It invokes the installed `sol-execbench` command with explicit paths (`scripts/run_dataset.py:247`, `scripts/run_dataset.py:280`).
4. It reads trace JSONL, derives summary metrics, optional timing evidence, AMD-native score reports, and execution closure records (`scripts/run_dataset.py:397`, `scripts/run_dataset.py:566`, `scripts/run_dataset.py:1033`).

**State Management:**
- Canonical run state lives in filesystem artifacts: staged temp directories, trace JSONL, and optional sidecars. Parent-process state is transient in `src/sol_execbench/cli/main.py`.
- Domain objects are Pydantic models or dataclasses; `Solution` is frozen and memoizes a content hash (`src/sol_execbench/core/data/solution.py:265`).
- The evaluation subprocess owns mutable GPU tensors, timing buffers, and reward-hack snapshots; that state is discarded when the subprocess exits (`src/sol_execbench/driver/templates/eval_driver.py`).

## Key Abstractions

**Definition:**
- Purpose: Single source of truth for operation metadata, symbolic axes, input/output tensors, and Python reference code.
- Examples: `src/sol_execbench/core/data/definition.py`, `docs/definition.md`, problem `definition.json` files.
- Pattern: Pydantic model with AST validation for the required top-level `run` reference function and input-name matching.

**Workload:**
- Purpose: Concrete axis values, input data descriptors, tolerance, and UUID for one executable case.
- Examples: `src/sol_execbench/core/data/workload.py`, problem `workload.jsonl` files.
- Pattern: Pydantic union of random, scalar, safetensors, and custom input specs.

**Solution / BuildSpec:**
- Purpose: Submitted implementation metadata, target ROCm hardware, language category, entry point, sources, and compile flags.
- Examples: `src/sol_execbench/core/data/solution.py`, example `solution.json` files under `examples/`.
- Pattern: Pydantic model with language-family validation, entry-point validation, source path security checks, and deterministic content hashing.

**Trace / Evaluation:**
- Purpose: Canonical benchmark output linking definition, workload, solution, status, environment, correctness, and performance.
- Examples: `src/sol_execbench/core/data/trace.py`, trace JSONL written by `src/sol_execbench/cli/main.py`.
- Pattern: Pydantic model with status-dependent validation for correctness/performance fields.

**ProblemPackager:**
- Purpose: Convert in-memory models into a staged executable bundle.
- Examples: `src/sol_execbench/driver/problem_packager.py`.
- Pattern: Command factory that writes files in `__init__`, returns compile/evaluate command arrays, and cleans the temp directory in `__del__` unless kept.

**Evaluation Driver Template:**
- Purpose: Isolated executable that imports user code, checks reward-hack defenses, evaluates all workloads, and prints strict JSONL.
- Examples: `src/sol_execbench/driver/templates/eval_driver.py`.
- Pattern: Generated script copied into staging; redirects noisy stdout to stderr and writes JSON only to the original stdout file descriptor.

**Toolchain Capability:**
- Purpose: Describe which ROCm tools can provide runtime, profiling, static, or derived-score evidence for specific artifacts/hardware.
- Examples: `src/sol_execbench/core/toolchain.py`.
- Pattern: Pydantic registry entries plus route decisions with bounded probes.

## Entry Points

**`sol-execbench`:**
- Location: `src/sol_execbench/cli/main.py`, exposed by `pyproject.toml`.
- Triggers: CLI invocation.
- Responsibilities: Evaluate one problem/solution pair, print contract/doctor/toolchain metadata, write trace and sidecar files.

**`sol-execbench-baseline`:**
- Location: `src/sol_execbench/cli/baseline.py`, exposed by `pyproject.toml`.
- Triggers: CLI invocation.
- Responsibilities: Compare trace JSONL files using baseline guardrails from `src/sol_execbench/core/baseline.py`.

**Dataset Runner:**
- Location: `scripts/run_dataset.py`.
- Triggers: `uv run scripts/run_dataset.py <downloaded-benchmark-dir>`.
- Responsibilities: Discover problems, construct solutions, invoke `sol-execbench`, summarize results, write dataset-level evidence.

**Report Scripts:**
- Location: `scripts/report_*.py`, `scripts/diff_matrix_reports.py`, `scripts/export_matrix_schema.py`, `scripts/inspect_dataset.py`.
- Triggers: Direct `uv run scripts/...` invocation.
- Responsibilities: Generate focused JSON/Markdown reports from core dataset, scoring, consistency, or matrix modules.

**Docker Wrapper:**
- Location: `scripts/run_docker.sh`, `docker/entrypoint.sh`, `docker/Dockerfile`.
- Triggers: `./scripts/run_docker.sh --build` or container startup.
- Responsibilities: Build/enter ROCm evaluation environment and configure runtime prerequisites.

## Architectural Constraints

- **Threading:** Parent orchestration is synchronous subprocess execution in `src/sol_execbench/cli/main.py`. Evaluation uses one Python process and explicitly checks thread-count increases as a reward-hack signal in `src/sol_execbench/driver/templates/eval_driver.py` and `src/sol_execbench/core/bench/reward_hack.py`.
- **Global state:** `src/sol_execbench/driver/templates/eval_driver.py` uses module-level loaded problem/config objects and reward-hack snapshots; `src/sol_execbench/core/bench/reward_hack.py` captures `torch.cuda.Event.elapsed_time` identity at import time; `src/sol_execbench/cli/main.py` has a module-level Rich `Console`.
- **Circular imports:** Public re-export modules `src/sol_execbench/__init__.py` and `src/sol_execbench/core/__init__.py` intentionally aggregate data/environment/toolchain APIs. New core modules should import concrete submodules (`src/sol_execbench/core/data/...`) instead of importing back from `sol_execbench.core` when they sit inside `core`.
- **User code isolation:** Submitted sources must flow through `ProblemPackager` and `eval_driver.py`; do not run submitted code in the parent CLI or dataset process.
- **Trace contract:** Canonical benchmark output is `Trace` JSONL from `src/sol_execbench/core/data/trace.py`. Optional diagnostics must remain sidecars or derived reports unless the public schema is intentionally changed.
- **ROCm-only language surface:** `BuildSpec` rejects CUDA/NVIDIA language and compile option names in `src/sol_execbench/core/data/solution.py`.

## Anti-Patterns

### Parent-Process User Code Import

**What happens:** Importing solution source directly from `src/sol_execbench/cli/`, `scripts/`, or test helpers bypasses staging, static source review, subprocess lifetime boundaries, and stdout isolation.
**Why it's wrong:** It lets submitted code mutate the orchestrator process, corrupt logs, or affect later evaluations.
**Do this instead:** Add behavior to `src/sol_execbench/driver/templates/eval_driver.py` or stage files through `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py`.

### Canonical Trace Overloading

**What happens:** Adding environment snapshots, profiling evidence, static artifacts, or derived AMD score fields directly into `Trace`.
**Why it's wrong:** `Trace` is the public compatibility contract; side evidence has different authority and can be unavailable without invalidating an evaluation.
**Do this instead:** Write sidecars through helpers in `src/sol_execbench/cli/main.py`, evidence models in `src/sol_execbench/core/bench/`, or derived reports in `src/sol_execbench/core/scoring/`.

### Importing From Aggregator Inside Core

**What happens:** A new module inside `src/sol_execbench/core/` imports from `sol_execbench.core` instead of a concrete sibling module.
**Why it's wrong:** Aggregator imports in `src/sol_execbench/core/__init__.py` increase circular-import risk as the core package grows.
**Do this instead:** Import from concrete modules such as `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/bench/timing.py`, or `src/sol_execbench/core/toolchain.py`.

## Error Handling

**Strategy:** Validate schemas early with Pydantic, convert user-facing CLI failures into `click.ClickException` or nonzero CLI exits, and encode per-workload execution failures as `Trace.evaluation.status`.

**Patterns:**
- Input file resolution failures are raised as `click.ClickException` in `src/sol_execbench/cli/main.py`.
- Compile subprocess failures print captured stdout/stderr and exit before evaluation in `src/sol_execbench/cli/main.py`.
- Evaluation subprocess errors become `Trace` statuses such as `INVALID_REFERENCE`, `RUNTIME_ERROR`, `COMPILE_ERROR`, `TIMEOUT`, or `REWARD_HACK` in `src/sol_execbench/core/data/trace.py`.
- Optional evidence failures are nonfatal and become skipped/failed sidecars or warnings in `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and `src/sol_execbench/core/bench/static_kernel_evidence.py`.

## Cross-Cutting Concerns

**Logging:** CLI status and tables use Rich via `console = Console(stderr=True)` in `src/sol_execbench/cli/main.py`. Evaluation driver redirects ordinary stdout to stderr and writes JSONL to the original stdout descriptor in `src/sol_execbench/driver/templates/eval_driver.py`.

**Validation:** Pydantic models validate public schemas in `src/sol_execbench/core/data/`; AST-based checks validate reference code and entry points in `src/sol_execbench/core/data/definition.py` and `src/sol_execbench/core/data/solution.py`.

**Authentication:** Not applicable. The project is a local benchmark/evaluation package with no application identity provider.

**Security:** Source path traversal is rejected in `SourceFile` (`src/sol_execbench/core/data/solution.py`), reward-hack patterns are blocked in `src/sol_execbench/core/bench/reward_hack.py`, dynamic C++ extension loading is blocked in Python solutions by `src/sol_execbench/driver/templates/eval_driver.py`, and secrets/datasets are excluded by repository policy.

---

*Architecture analysis: 2026-05-31*
