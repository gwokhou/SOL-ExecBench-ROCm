<!-- refreshed: 2026-05-22 -->
# Architecture

**Analysis Date:** 2026-05-22

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
├───────────────────────────────┬─────────────────────────────┤
│ Benchmark CLI                 │ Baseline CLI                │
│ `src/sol_execbench/cli/main.py` │ `src/sol_execbench/cli/baseline.py` │
└───────────────┬───────────────┴──────────────┬──────────────┘
                │                              │
                ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Core Layer                          │
├───────────────────┬───────────────────┬─────────────────────┤
│ Data contracts    │ Benchmark utils   │ Reporting/scoring   │
│ `src/sol_execbench/core/data/` │ `src/sol_execbench/core/bench/` │ `src/sol_execbench/core/scoring/` │
└───────────────┬───┴───────────────┬───┴──────────────┬──────┘
                │                   │                  │
                ▼                   ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                       Driver Layer                          │
│ `src/sol_execbench/driver/problem_packager.py`               │
│ `src/sol_execbench/driver/templates/`                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Staged Subprocess Runtime                 │
│ `definition.json`, `workload.jsonl`, `solution.json`,        │
│ `config.json`, `eval_driver.py`, optional `build_ext.py`     │
└─────────────────────────────────────────────────────────────┘
```

SOL ExecBench ROCm is a layered Python package. The CLI loads JSON inputs into typed contracts, `ProblemPackager` materializes a staging directory, generated driver scripts run candidate code in a subprocess, and the subprocess emits JSONL `Trace` records that the CLI parses and reports.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Benchmark CLI | Resolve problem paths, load schemas, create staging directory, run compile/eval subprocesses, print or write traces. | `src/sol_execbench/cli/main.py` |
| Baseline CLI | Compare candidate trace JSONL against baseline trace JSONL and render text or JSON. | `src/sol_execbench/cli/baseline.py` |
| Public exports | Re-export data models and JSON helpers for package consumers. | `src/sol_execbench/__init__.py`, `src/sol_execbench/core/__init__.py` |
| Data contracts | Define Pydantic schemas for definitions, workloads, solutions, traces, dtypes, and JSON utilities. | `src/sol_execbench/core/data/` |
| Benchmark execution helpers | Generate inputs, allocate outputs, normalize outputs, check correctness, time GPU functions, detect reward-hack behavior, and manage clock/profiler utilities. | `src/sol_execbench/core/bench/` |
| Reporting | Summarize trace runs and build derived evidence reports. | `src/sol_execbench/core/reporting.py` |
| Baseline comparison | Load trace files and classify candidate latency against best matching baseline latency. | `src/sol_execbench/core/baseline.py` |
| AMD scoring | Build AMD-native SOL bound artifacts and guarded suite score reports. | `src/sol_execbench/core/scoring/` |
| Driver packager | Write normalized inputs and source files into a staging directory and return subprocess commands. | `src/sol_execbench/driver/problem_packager.py` |
| Evaluation driver template | Run reference and user code per workload, enforce guardrails, time solutions, and emit `Trace` JSONL. | `src/sol_execbench/driver/templates/eval_driver.py` |
| Native build template | Compile HIP/C++ sources through PyTorch's extension API into `benchmark_kernel.so`. | `src/sol_execbench/driver/templates/build_ext.py` |

## Pattern Overview

**Overall:** Layered CLI application with typed data contracts and subprocess isolation.

**Key Characteristics:**
- Keep public benchmark semantics in Pydantic models under `src/sol_execbench/core/data/`; CLI and driver code should consume these models instead of manipulating raw dictionaries.
- Keep user solution execution out of the CLI process. Candidate sources are copied to staging and imported by `src/sol_execbench/driver/templates/eval_driver.py`.
- Treat ROCm as the execution target while retaining PyTorch's HIP-backed `torch.cuda` device API where PyTorch ROCm exposes it.
- Keep native HIP/C++ compilation as a separate staged phase through `src/sol_execbench/driver/templates/build_ext.py`.

## Layers

**CLI Layer:**
- Purpose: User-facing commands, path resolution, subprocess orchestration, and console output.
- Location: `src/sol_execbench/cli/`
- Contains: Click commands, Rich table rendering, trace JSONL output handling.
- Depends on: `src/sol_execbench/core/`, `src/sol_execbench/driver/`
- Used by: `pyproject.toml` script entry points `sol-execbench` and `sol-execbench-baseline`.

**Core Data Layer:**
- Purpose: Public schema contracts and serialization helpers.
- Location: `src/sol_execbench/core/data/`
- Contains: `Definition`, `Workload`, `Solution`, `Trace`, dtype conversion, shape expression resolution, JSON/JSONL helpers.
- Depends on: Pydantic, Python standard library; dtype helpers lazily import PyTorch for torch dtype mapping.
- Used by: CLI loading, driver templates, benchmark utilities, scoring, tests, examples.

**Core Benchmark Layer:**
- Purpose: Runtime operations used by the staged evaluation driver.
- Location: `src/sol_execbench/core/bench/`
- Contains: Input generation in `src/sol_execbench/core/bench/io.py`, correctness in `src/sol_execbench/core/bench/correctness.py`, timing in `src/sol_execbench/core/bench/timing.py`, reward-hack checks in `src/sol_execbench/core/bench/reward_hack.py`, clock and profiler utilities.
- Depends on: PyTorch ROCm APIs, safetensors when safetensors inputs are used, ROCm command-line tools for clock/profiler paths.
- Used by: `src/sol_execbench/driver/templates/eval_driver.py`, diagnostics/scoring tests.

**Driver Layer:**
- Purpose: Convert typed benchmark inputs into a temporary filesystem runtime.
- Location: `src/sol_execbench/driver/`
- Contains: `ProblemPackager`, templates copied into staging, offload architecture detection for HIP builds.
- Depends on: Core data/config models, subprocess, filesystem utilities.
- Used by: `src/sol_execbench/cli/main.py`, `tests/sol_execbench/driver/`, `tests/examples/test_examples.py`.

**Scoring and Reporting Layer:**
- Purpose: Summarize traces, compare baselines, and produce guarded AMD-native scoring evidence.
- Location: `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/scoring/`
- Contains: Baseline comparison dataclasses, AMD SOL bound estimates, score guardrails, suite reports.
- Depends on: `Trace`, `Definition`, workload metadata, scoring guardrail helpers.
- Used by: `src/sol_execbench/cli/baseline.py`, `scripts/run_dataset.py`, scoring/reporting tests.

## Data Flow

### Primary Benchmark Path

1. The Click command receives either a problem directory or explicit JSON paths (`src/sol_execbench/cli/main.py:180`, `src/sol_execbench/cli/main.py:230`).
2. The CLI resolves `definition.json`, `workload.jsonl`, optional `config.json`, and `solution.json` (`src/sol_execbench/cli/main.py:83`).
3. JSON files are parsed into `Definition`, `Workload`, `Solution`, and `BenchmarkConfig` (`src/sol_execbench/cli/main.py:52`, `src/sol_execbench/cli/main.py:56`, `src/sol_execbench/cli/main.py:65`, `src/sol_execbench/cli/main.py:77`).
4. `ProblemPackager` writes normalized JSON files and source contents into a temporary staging directory (`src/sol_execbench/cli/main.py:286`, `src/sol_execbench/driver/problem_packager.py:91`).
5. HIP/C++ solutions run a compile subprocess using `build_ext.py`; Python and Triton solutions skip compile (`src/sol_execbench/cli/main.py:301`, `src/sol_execbench/driver/problem_packager.py:169`).
6. `ProblemPackager.execute()` stages `eval_driver.py` and returns `["python", "eval_driver.py"]` (`src/sol_execbench/driver/problem_packager.py:202`).
7. The CLI runs the evaluation subprocess with `PYTORCH_ALLOC_CONF=expandable_segments:True` (`src/sol_execbench/cli/main.py:335`).
8. The evaluation driver loads staged schemas, reference code, solution code, workloads, and config (`src/sol_execbench/driver/templates/eval_driver.py:90`, `src/sol_execbench/driver/templates/eval_driver.py:160`, `src/sol_execbench/driver/templates/eval_driver.py:164`).
9. For each workload, the driver generates inputs, runs reference and candidate functions, checks reward-hack defenses, validates shape/dtype/numerics, times the candidate, and emits one `Trace` JSON object (`src/sol_execbench/driver/templates/eval_driver.py:292`, `src/sol_execbench/driver/templates/eval_driver.py:376`).
10. The CLI parses JSONL stdout into `Trace` models, writes optional output, renders a table, and exits `0` only when all traces pass (`src/sol_execbench/driver/problem_packager.py:228`, `src/sol_execbench/cli/main.py:351`, `src/sol_execbench/cli/main.py:379`).

### Native HIP/C++ Compile Path

1. `ProblemPackager._is_cpp` checks whether any solution language is a native ROCm language (`src/sol_execbench/driver/problem_packager.py:43`, `src/sol_execbench/driver/problem_packager.py:130`).
2. `ProblemPackager.compile()` injects `--offload-arch=<gfx>` flags from explicit hardware or local ROCm tool detection when flags are absent (`src/sol_execbench/driver/problem_packager.py:132`, `src/sol_execbench/driver/problem_packager.py:169`).
3. `build_ext.py` validates `solution.json`, collects staged `.hip`/C/C++ files, and calls `torch.utils.cpp_extension.load()` (`src/sol_execbench/driver/templates/build_ext.py:21`, `src/sol_execbench/driver/templates/build_ext.py:35`).
4. The compiled module is normalized to `benchmark_kernel.so` in the staging directory (`src/sol_execbench/driver/templates/build_ext.py:53`).
5. The evaluation driver imports `benchmark_kernel.so` for native ROCm languages (`src/sol_execbench/driver/templates/eval_driver.py:245`).

### Baseline Comparison Path

1. `sol-execbench-baseline` receives candidate and baseline trace JSONL file paths (`src/sol_execbench/cli/baseline.py:21`, `src/sol_execbench/cli/baseline.py:70`).
2. `load_trace_jsonl()` parses each line into `Trace` models (`src/sol_execbench/core/baseline.py:51`).
3. `compare_trace_baselines()` groups passed baseline traces by `(definition, workload.uuid)` and keeps the lowest latency (`src/sol_execbench/core/baseline.py:65`).
4. Results are rendered as text with `format_baseline_comparison()` or JSON with `comparison_to_json()` (`src/sol_execbench/core/baseline.py:124`, `src/sol_execbench/core/baseline.py:165`).

**State Management:**
- Runtime state is filesystem-scoped to a temporary staging directory created in `src/sol_execbench/cli/main.py:286`.
- Benchmark configuration is a dataclass (`src/sol_execbench/core/bench/config/benchmark_config.py:24`) serialized to `config.json` by `ProblemPackager`.
- Model objects are immutable where behavioral identity matters; `Solution` is frozen and memoizes a content hash (`src/sol_execbench/core/data/solution.py:265`).
- The evaluation driver uses module-level variables because it is generated as a one-shot script executed in a subprocess (`src/sol_execbench/driver/templates/eval_driver.py`).

## Key Abstractions

**Definition:**
- Purpose: Formal benchmark problem contract with axes, input/output tensor specs, and reference Python code.
- Examples: `src/sol_execbench/core/data/definition.py:136`, `examples/pytorch/linear_backward/definition.json`
- Pattern: Pydantic model with validators for reference code, parameter matching, axis references, and tensor names.

**Workload:**
- Purpose: Concrete axis values, input source descriptors, UUID, and numerical tolerance for one benchmark run.
- Examples: `src/sol_execbench/core/data/workload.py:102`, `examples/pytorch/linear_backward/workload.jsonl`
- Pattern: Pydantic discriminated-style union by `type` fields for random, scalar, safetensors, and custom inputs.

**Solution:**
- Purpose: Candidate implementation metadata, source files, language categories, target hardware, entry point, and compile options.
- Examples: `src/sol_execbench/core/data/solution.py:265`, `examples/hip_cpp/rmsnorm/solution_hip.json`
- Pattern: Frozen Pydantic model with source path validation, ROCm language validation, and deterministic content hashing.

**Trace:**
- Purpose: Canonical JSONL output linking a definition, workload, optional solution name, and optional evaluation result.
- Examples: `src/sol_execbench/core/data/trace.py:176`, `docs/trace.md`
- Pattern: Pydantic model with status-specific validation for correctness and performance fields.

**ProblemPackager:**
- Purpose: Staging adapter between typed in-process models and subprocess files/commands.
- Examples: `src/sol_execbench/driver/problem_packager.py:91`, `tests/sol_execbench/driver/test_problem_packager.py`
- Pattern: Constructor writes base staging files; `compile()` and `execute()` add phase-specific templates and return command arrays.

**Evaluation Driver Template:**
- Purpose: Self-contained subprocess script for importing user code, running workloads, and emitting strict JSON traces.
- Examples: `src/sol_execbench/driver/templates/eval_driver.py`
- Pattern: Generated script with early stdout redirection, module-level staged state, and helper functions for emitting traces and detecting benchmark abuse.

**ShiftingMemoryPoolAllocator:**
- Purpose: Provide unique tensor data pointers across timing iterations without full data duplication.
- Examples: `src/sol_execbench/core/bench/io.py:519`, `src/sol_execbench/core/bench/timing.py:161`
- Pattern: Pre-allocate memory pools and provide fresh strided views via `get_unique_args()`.

## Entry Points

**Benchmark CLI:**
- Location: `src/sol_execbench/cli/main.py:230`
- Triggers: `sol-execbench` script from `pyproject.toml`.
- Responsibilities: Validate inputs, run compile/evaluation subprocesses, parse `Trace` JSONL, and determine process exit code.

**Baseline CLI:**
- Location: `src/sol_execbench/cli/baseline.py:70`
- Triggers: `sol-execbench-baseline` script from `pyproject.toml`.
- Responsibilities: Load trace JSONL, compare candidate latencies to baselines, render comparison output.

**Dataset Runner:**
- Location: `scripts/run_dataset.py`
- Triggers: `uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5`
- Responsibilities: Batch over benchmark assets and drive package scoring/report generation.

**Docker Entrypoint:**
- Location: `docker/entrypoint.sh`
- Triggers: `./scripts/run_docker.sh --build` or container startup.
- Responsibilities: Prepare GPU evaluation environment and clock/runtime setup.

**Package API:**
- Location: `src/sol_execbench/__init__.py`, `src/sol_execbench/core/__init__.py`
- Triggers: `import sol_execbench`
- Responsibilities: Export schema models, config helpers, and JSON utilities.

## Architectural Constraints

- **Threading:** Main CLI work is synchronous. Evaluation runs in a subprocess. `eval_driver.py` samples `threading.active_count()` to detect thread injection around timing (`src/sol_execbench/driver/templates/eval_driver.py:514`).
- **Global state:** `eval_driver.py` intentionally uses module-level staged state such as `STAGING_DIR`, `definition`, `workloads`, `_solution`, `_device`, and `bench_config` because it is a generated one-shot script (`src/sol_execbench/driver/templates/eval_driver.py:86`).
- **Circular imports:** No required circular import chains detected in `src/sol_execbench/`; package-level `__init__.py` files re-export core models for public API convenience.
- **Subprocess boundary:** User code must run through staged subprocess scripts. Do not import candidate solution modules in `src/sol_execbench/cli/main.py`.
- **Public schema stability:** Public contracts live in `src/sol_execbench/core/data/`; schema changes need corresponding docs and tests under `docs/` and `tests/sol_execbench/`.
- **ROCm API naming:** PyTorch ROCm still exposes HIP-backed device APIs through `torch.cuda`; do not rename these calls mechanically when they represent the PyTorch ROCm device API (`src/sol_execbench/core/bench/timing.py:67`).

## Anti-Patterns

### Importing User Code In The CLI

**What happens:** Candidate solution modules are imported directly from `src/sol_execbench/cli/main.py`.
**Why it's wrong:** It bypasses subprocess isolation, stdout redirection, reward-hack checks, compile separation, and staging cleanup.
**Do this instead:** Add runtime logic to `src/sol_execbench/driver/templates/eval_driver.py` and stage it through `src/sol_execbench/driver/problem_packager.py`.

### Mutating Raw Schema Dictionaries Past Load Boundaries

**What happens:** Code passes unvalidated dictionaries through benchmark execution after JSON parsing.
**Why it's wrong:** It skips validators in `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, and `src/sol_execbench/core/data/trace.py`.
**Do this instead:** Convert JSON to Pydantic models at the boundary, then use model methods such as `Definition.get_resolved_axes_values()` and `Solution.get_entry_symbol()`.

### Adding CUDA/NVIDIA Language Paths As Active Backends

**What happens:** New code accepts CUDA-specific language values or compile option keys in solution specs.
**Why it's wrong:** `BuildSpec` rejects legacy CUDA/NVIDIA language values and maps ROCm-native replacements (`src/sol_execbench/core/data/solution.py:132`).
**Do this instead:** Add ROCm language/library categories to `SupportedLanguages`, `_CPP_LANGUAGES`, and `_NATIVE_ROCM_LANGUAGES` when needed (`src/sol_execbench/core/data/solution.py:28`, `src/sol_execbench/driver/problem_packager.py:43`, `src/sol_execbench/driver/templates/eval_driver.py:245`).

## Error Handling

**Strategy:** Validate early with Pydantic and Click, convert per-workload runtime failures into `Trace` statuses inside the evaluation driver, and reserve process failure for staging/evaluation failures that produce no traces.

**Patterns:**
- Use `click.ClickException` for missing CLI inputs (`src/sol_execbench/cli/main.py:83`, `src/sol_execbench/cli/main.py:262`).
- Use Pydantic validators for schema invariants (`src/sol_execbench/core/data/definition.py:170`, `src/sol_execbench/core/data/solution.py:145`, `src/sol_execbench/core/data/trace.py:127`).
- Emit `EvaluationStatus.RUNTIME_ERROR`, `INVALID_REFERENCE`, `INCORRECT_*`, or `REWARD_HACK` traces from `eval_driver.py` for workload-local failures (`src/sol_execbench/driver/templates/eval_driver.py:306`, `src/sol_execbench/driver/templates/eval_driver.py:416`).
- Exit the CLI with nonzero status when compilation fails, no traces are produced, or not all traces pass (`src/sol_execbench/cli/main.py:317`, `src/sol_execbench/cli/main.py:355`, `src/sol_execbench/cli/main.py:379`).

## Cross-Cutting Concerns

**Logging:** CLI user output uses Rich in `src/sol_execbench/cli/main.py`; evaluation driver redirects non-JSON stdout to stderr so stdout stays parseable JSONL (`src/sol_execbench/driver/templates/eval_driver.py:41`).
**Validation:** Pydantic models in `src/sol_execbench/core/data/` own schema validation; tests in `tests/sol_execbench/core/data/` and public contract tests guard this behavior.
**Authentication:** Not applicable; the package is a local benchmark CLI with no built-in auth provider.
**Security:** Source path traversal is rejected in `SourceFile` (`src/sol_execbench/core/data/solution.py:82`); reward-hack checks inspect sources, monkey patches, lazy outputs, thread injection, and critical function identity (`src/sol_execbench/core/bench/reward_hack.py`).
**Hardware/ROCm detection:** Local gfx detection uses `rocm_agent_enumerator` or `rocminfo` in `src/sol_execbench/driver/problem_packager.py`; diagnostics and profiler readiness live in `src/sol_execbench/core/diagnostics.py` and `src/sol_execbench/core/bench/rocm_profiler.py`.

---

*Architecture analysis: 2026-05-22*
