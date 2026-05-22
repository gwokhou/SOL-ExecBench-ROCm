<!-- refreshed: 2026-05-22 -->
# Architecture

**Analysis Date:** 2026-05-22

## System Overview

```text
+-------------------------------------------------------------+
|                         CLI Layer                            |
| `src/sol_execbench/cli/main.py`  `src/sol_execbench/cli/baseline.py` |
+---------------------+---------------------+-----------------+
                      |
                      v
+-------------------------------------------------------------+
|                   Public Core Model Surface                  |
| `src/sol_execbench/core/__init__.py`                         |
| Pydantic schemas, config, trace/result objects               |
+---------------------+---------------------+-----------------+
                      |
                      v
+---------------------+---------------------+-----------------+
|    Staging Driver    |       Bench Runtime      | Reporting   |
| `driver/`            | `core/bench/`            | `core/*.py`  |
| ProblemPackager      | input/correctness/timing | summaries    |
+----------+----------+-----------+-------------+-------------+
           |                      |
           v                      v
+-------------------------------------------------------------+
|              Subprocess Evaluation Sandbox                   |
| `src/sol_execbench/driver/templates/build_ext.py`            |
| `src/sol_execbench/driver/templates/eval_driver.py`          |
+---------------------+---------------------+-----------------+
                      |
                      v
+-------------------------------------------------------------+
|         Filesystem Artifacts and Trace JSONL Output           |
| temp staging dirs, `definition.json`, `workload.jsonl`,       |
| `solution.json`, `config.json`, `benchmark_kernel.so`, traces |
+-------------------------------------------------------------+
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Main benchmark CLI | Parse CLI options, load problem files into models, create staging directory, invoke compile/evaluate subprocesses, render trace results. | `src/sol_execbench/cli/main.py` |
| Baseline CLI | Compare candidate trace JSONL files against one or more baseline trace JSONL files. | `src/sol_execbench/cli/baseline.py` |
| Public package exports | Re-export the schema and config types used by the CLI, driver, tests, and examples. | `src/sol_execbench/core/__init__.py`, `src/sol_execbench/__init__.py` |
| Schema models | Define benchmark contracts for definitions, workloads, solutions, traces, evaluation metrics, dtypes, shapes, and JSON helpers. | `src/sol_execbench/core/data/` |
| Benchmark config | Hold runtime knobs such as seeds, warmup/iteration counts, reference timing, and clock-lock requirements. | `src/sol_execbench/core/bench/config/benchmark_config.py` |
| Problem packager | Materialize validated models and submitted sources into a temporary staging directory and return subprocess commands. | `src/sol_execbench/driver/problem_packager.py` |
| Build template | Compile HIP/C++ sources into `benchmark_kernel.so` through `torch.utils.cpp_extension.load`. | `src/sol_execbench/driver/templates/build_ext.py` |
| Evaluation template | Load reference and submitted code, generate inputs, run correctness checks, time kernels, apply reward-hack defenses, and emit `Trace` JSONL. | `src/sol_execbench/driver/templates/eval_driver.py` |
| Bench I/O | Generate random/custom/safetensors inputs, normalize return-value outputs, allocate DPS outputs, and provide unique memory views for timing. | `src/sol_execbench/core/bench/io.py` |
| Correctness | Seed execution and compute numerical error metrics against workload tolerances. | `src/sol_execbench/core/bench/correctness.py` |
| Timing | Measure ROCm kernels using PyTorch HIP-backed `torch.cuda.Event` timing with cache clearing and shifted tensor addresses. | `src/sol_execbench/core/bench/timing.py` |
| Reward-hack checks | Detect timing monkey-patches, lazy output proxies, thread injection, and eval-driver integrity changes. | `src/sol_execbench/core/bench/reward_hack.py` |
| Diagnostics/reporting | Provide internal ROCm readiness diagnostics and derived summaries without changing trace schema. | `src/sol_execbench/core/diagnostics.py`, `src/sol_execbench/core/reporting.py` |

## Pattern Overview

**Overall:** CLI-orchestrated, schema-first benchmark runner with generated staging scripts and subprocess isolation.

**Key Characteristics:**
- Treat `Definition`, `Workload`, `Solution`, and `Trace` Pydantic models in `src/sol_execbench/core/data/` as the public benchmark contract.
- Keep user code out of the parent CLI process. The CLI stages files with `ProblemPackager`, then runs `build_ext.py` and `eval_driver.py` under `subprocess.run`.
- Use filesystem staging as the handoff boundary: staged JSON models, source files, optional compiled `benchmark_kernel.so`, and stdout trace JSONL are the integration format.
- Keep ROCm compatibility behind existing PyTorch CUDA-named APIs where PyTorch exposes HIP behavior, especially `torch.cuda.Event`, `torch.cuda.synchronize`, and `torch.utils.cpp_extension`.

## Layers

**CLI Layer:**
- Purpose: Own user interaction, option parsing, problem-file resolution, output formatting, and process exit codes.
- Location: `src/sol_execbench/cli/`
- Contains: `sol-execbench` main CLI in `src/sol_execbench/cli/main.py`; `sol-execbench-baseline` in `src/sol_execbench/cli/baseline.py`.
- Depends on: `src/sol_execbench/core/`, `src/sol_execbench/driver/problem_packager.py`, Click, Rich, subprocess.
- Used by: Package scripts configured in `pyproject.toml` and CLI e2e tests in `tests/sol_execbench/test_e2e.py`.

**Data Contract Layer:**
- Purpose: Validate and serialize every public benchmark artifact.
- Location: `src/sol_execbench/core/data/`
- Contains: `Definition`, `Workload`, `Solution`, `Trace`, JSON helpers, dtype conversion, and shape resolution.
- Depends on: Pydantic, Python AST parsing, PyTorch dtype APIs.
- Used by: CLI loaders, `ProblemPackager`, generated templates, tests, examples, scoring, reporting, and baseline comparison.

**Driver/Staging Layer:**
- Purpose: Convert validated in-memory models into executable staging directories.
- Location: `src/sol_execbench/driver/`
- Contains: `ProblemPackager`, ROCm gfx detection helpers, template files.
- Depends on: `src/sol_execbench/core/`, filesystem writes to temp staging dirs, ROCm tools `rocm_agent_enumerator` and `rocminfo` for LOCAL target expansion.
- Used by: Main CLI, tests under `tests/sol_execbench/driver/`, example and e2e tests.

**Evaluation Runtime Layer:**
- Purpose: Execute reference and user functions, enforce benchmark semantics, and emit canonical trace JSONL.
- Location: `src/sol_execbench/driver/templates/eval_driver.py` and helpers in `src/sol_execbench/core/bench/`
- Contains: Reference import, user import/native module load, input generation, output normalization, correctness, timing, reward-hack checks, trace emission.
- Depends on: PyTorch ROCm runtime, safetensors for blob-backed workloads, staged files.
- Used by: Subprocesses launched from the CLI and direct template tests in `tests/sol_execbench/driver/test_eval_driver.py`.

**Reporting and Scoring Layer:**
- Purpose: Summarize or compare existing trace objects without changing the trace JSONL schema.
- Location: `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/sol_score.py`
- Contains: Baseline comparison dataclasses, trace summary dataclasses, score helper.
- Depends on: `Trace` and `EvaluationStatus` models.
- Used by: Baseline CLI and tests under `tests/sol_execbench/`.

**Documentation and Examples Layer:**
- Purpose: Preserve schema, ROCm migration, compatibility, and runnable example guidance.
- Location: `docs/`, `examples/`, `tests/examples/`
- Contains: Schema docs, ROCm support docs, example problem directories with `definition.json`, `workload.jsonl`, `solution_*.json`, and source files.
- Depends on: Core schema and driver behavior.
- Used by: Docs guardrail tests and example workflow tests.

## Data Flow

### Primary Request Path

1. User invokes `sol-execbench` via `pyproject.toml` script `sol_execbench.cli:cli`; Click options are defined in `src/sol_execbench/cli/main.py:180`.
2. CLI resolves either `problem_dir` or explicit `--definition`, `--workload`, and `--solution` paths in `src/sol_execbench/cli/main.py:252`.
3. CLI loads JSON into `Definition`, `Workload`, `Solution`, and `BenchmarkConfig` models in `src/sol_execbench/cli/main.py:269`.
4. CLI creates a temp staging directory and instantiates `ProblemPackager` in `src/sol_execbench/cli/main.py:286`.
5. `ProblemPackager.__init__` writes `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, and submitted sources to staging in `src/sol_execbench/driver/problem_packager.py:112`.
6. For native ROCm/C++ languages, CLI runs the `ProblemPackager.compile()` command from `src/sol_execbench/driver/problem_packager.py:175`; the staged `build_ext.py` compiles `benchmark_kernel.so` in `src/sol_execbench/driver/templates/build_ext.py`.
7. CLI stages and runs `eval_driver.py` through `ProblemPackager.execute()` in `src/sol_execbench/driver/problem_packager.py:203` and `src/sol_execbench/cli/main.py:332`.
8. `eval_driver.py` parses staged models, imports reference code, imports submitted Python code or compiled `benchmark_kernel.so`, and selects device in `src/sol_execbench/driver/templates/eval_driver.py:158` through `src/sol_execbench/driver/templates/eval_driver.py:250`.
9. For each workload, `eval_driver.py` resolves axes, loads safetensors if needed, generates inputs, runs reference and user functions, checks shape/dtype/numerics, times the user function, optionally times the reference, and emits one `Trace` JSON object in `src/sol_execbench/driver/templates/eval_driver.py:380` through `src/sol_execbench/driver/templates/eval_driver.py:656`.
10. CLI parses stdout JSONL into `Trace` models using `ProblemPackager.convert_stdout_to_traces()` in `src/sol_execbench/driver/problem_packager.py:228`, then writes output JSONL or renders a Rich table in `src/sol_execbench/cli/main.py`.

### HIP/C++ Compile Flow

1. `Solution.spec.languages` is validated as native ROCm or Python/Triton but not both in `src/sol_execbench/core/data/solution.py:210`.
2. `ProblemPackager._is_cpp` classifies native ROCm languages in `src/sol_execbench/driver/problem_packager.py:127`.
3. `ProblemPackager._inject_offload_arch_flags()` adds `--offload-arch=<gfx>` flags from explicit `target_hardware` values or local ROCm tool detection in `src/sol_execbench/driver/problem_packager.py:131`.
4. `build_ext.py` reads staged `solution.json`, collects top-level HIP/C/C++ files in the staging directory, passes `hip_cflags`, `cflags`, and `ld_flags` into `torch.utils.cpp_extension.load`, and normalizes the output shared object name to `benchmark_kernel.so`.
5. `eval_driver.py` imports `benchmark_kernel.so` as module `benchmark_kernel` and resolves the entry symbol in `src/sol_execbench/driver/templates/eval_driver.py:218`.

### Python/Triton Evaluation Flow

1. `Solution.spec.entry_point` must use `<file_path>::<function_name>` and Python solutions require a `.py` entry file in `src/sol_execbench/core/data/solution.py:194` and `src/sol_execbench/core/data/solution.py:257`.
2. `ProblemPackager._write_sources()` writes all `Solution.sources` under staging paths in `src/sol_execbench/driver/problem_packager.py:168`.
3. `eval_driver.py` inserts the staging directory on `sys.path`, blocks `torch.utils.cpp_extension.load/load_inline` for Python solutions, imports the staged module, and resolves the function in `src/sol_execbench/driver/templates/eval_driver.py:226`.

### Baseline Comparison Flow

1. `sol-execbench-baseline` loads candidate and baseline trace JSONL paths from Click options in `src/sol_execbench/cli/baseline.py`.
2. `load_trace_jsonl()` validates each row as a `Trace` in `src/sol_execbench/core/baseline.py`.
3. `compare_trace_baselines()` groups passed baseline traces by `(definition, workload.uuid)` and classifies candidate latency as `WIN`, `PARITY`, `LOSS`, `NO_CANDIDATE`, or `NO_BASELINE` in `src/sol_execbench/core/baseline.py`.
4. CLI renders text or JSON with `format_baseline_comparison()` or `comparison_to_json()` in `src/sol_execbench/core/baseline.py`.

**State Management:**
- Long-lived application state is intentionally minimal. The parent CLI keeps in-memory validated models and a temporary staging path for one run.
- Evaluation state is process-local inside `eval_driver.py`: parsed models, imported functions, integrity snapshots, per-workload tensors, and clock status.
- Benchmark reproducibility state is in `BenchmarkConfig` plus `set_seed()` in `src/sol_execbench/core/bench/correctness.py`.
- Temporary files are owned by `ProblemPackager`; `ProblemPackager.__del__` removes the staging directory unless `keep_output_dir` is true in `src/sol_execbench/driver/problem_packager.py:123`.

## Key Abstractions

**Definition:**
- Purpose: Formal workload contract: axes, tensor specs, reference implementation, optional custom input entrypoint, and shape/dtype helpers.
- Examples: `src/sol_execbench/core/data/definition.py`, `docs/definition.md`, `examples/*/*/definition.json`
- Pattern: Pydantic model with AST validators and cached shape/dtype properties. Add schema changes here first and update docs/tests.

**Workload:**
- Purpose: Concrete axis values, input generation strategy, workload UUID, and numerical tolerance.
- Examples: `src/sol_execbench/core/data/workload.py`, `examples/*/*/workload.jsonl`
- Pattern: Pydantic discriminated-style unions by `type` for random, scalar, safetensors, and custom inputs.

**Solution / BuildSpec / SourceFile:**
- Purpose: Submitted implementation metadata, source payloads, language/hardware target, entrypoint, DPS mode, and compile flags.
- Examples: `src/sol_execbench/core/data/solution.py`, `examples/*/*/solution_*.json`
- Pattern: Frozen Pydantic model with source-path validation, entrypoint validation, legacy CUDA value rejection, and deterministic content hash.

**Trace / Evaluation:**
- Purpose: Canonical result format emitted by evaluation subprocesses and consumed by reporting, scoring, and baseline comparison.
- Examples: `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/reporting.py`
- Pattern: Pydantic model with status-dependent metric validation. Preserve this schema for public compatibility.

**ProblemPackager:**
- Purpose: Bridge between validated objects and executable staging files.
- Examples: `src/sol_execbench/driver/problem_packager.py`, `tests/sol_execbench/driver/test_problem_packager.py`
- Pattern: Writes files once, returns simple `["python", "..."]` commands, and parses JSONL stdout back into `Trace`.

**Evaluation Driver Template:**
- Purpose: Self-contained script copied into staging and executed in a subprocess.
- Examples: `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Pattern: Use staged JSON files and imported `sol_execbench.core` helpers. Emit only strict JSON trace rows to original stdout; redirect library noise to stderr.

**ShiftingMemoryPoolAllocator:**
- Purpose: Prevent timing artifacts from repeated identical tensor addresses.
- Examples: `src/sol_execbench/core/bench/io.py:519`, `src/sol_execbench/core/bench/timing.py:161`
- Pattern: Pre-allocate pools and return shifted as-strided views for each warmup/timed iteration.

## Entry Points

**Main CLI:**
- Location: `src/sol_execbench/cli/main.py`
- Triggers: `sol-execbench` package script in `pyproject.toml`.
- Responsibilities: Load problem artifacts, package staging files, run compile/evaluation subprocesses, parse traces, write output, set exit code.

**Baseline CLI:**
- Location: `src/sol_execbench/cli/baseline.py`
- Triggers: `sol-execbench-baseline` package script in `pyproject.toml`.
- Responsibilities: Load trace JSONL files, compare candidate latencies against baselines, render text/JSON output.

**Generated Build Script:**
- Location: `src/sol_execbench/driver/templates/build_ext.py`
- Triggers: `ProblemPackager.compile()` and CLI subprocess execution.
- Responsibilities: Compile staged HIP/C/C++ sources and produce `benchmark_kernel.so`.

**Generated Evaluation Script:**
- Location: `src/sol_execbench/driver/templates/eval_driver.py`
- Triggers: `ProblemPackager.execute()` and CLI subprocess execution.
- Responsibilities: Execute all workloads and emit canonical trace JSONL.

**Dataset Runner Script:**
- Location: `scripts/run_dataset.py`
- Triggers: Manual command `uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5`.
- Responsibilities: Batch-run benchmark problem directories through the installed CLI-style workflow.

**Docker Entrypoint:**
- Location: `docker/entrypoint.sh`
- Triggers: Container startup through `./scripts/run_docker.sh`.
- Responsibilities: Prepare the ROCm/GPU evaluation environment and clock-lock behavior used by benchmark runs.

## Architectural Constraints

- **Threading:** The benchmark path is synchronous in the parent CLI. User code may initialize JIT infrastructure threads during correctness warmup, but new thread injection during timing is rejected by `check_thread_injection()` in `src/sol_execbench/core/bench/reward_hack.py`.
- **Global state:** `eval_driver.py` uses module-level runtime state for staged models, imported functions, device, integrity snapshot, and per-workload buffers. Keep helper functions pure where practical because user code integrity checks snapshot named globals in `src/sol_execbench/driver/templates/eval_driver.py:185`.
- **Subprocess boundary:** Evaluation and compilation happen in child Python processes. Do not import or execute submitted solution code in `src/sol_execbench/cli/main.py`.
- **Public schema stability:** `Definition`, `Workload`, `Solution`, and `Trace` are public contracts. Additive changes need docs and guardrail tests under `tests/sol_execbench/`.
- **ROCm-only target:** Native language values are HIP/ROCm-specific; legacy CUDA/NVIDIA schema values are rejected in `src/sol_execbench/core/data/solution.py:158`.
- **Device naming:** PyTorch ROCm still uses `torch.cuda.*`; use these APIs consistently for HIP-backed devices instead of introducing parallel CUDA/ROCm wrappers unless a real abstraction is needed.
- **Filesystem staging:** Source paths inside `Solution.sources` must remain relative and cannot contain `..`; staged files are the execution boundary.
- **Circular imports:** Not detected in the package structure. Keep dependencies flowing from CLI to core/driver and from templates to core helpers; avoid importing CLI code from core modules.

## Anti-Patterns

### Executing User Code In The CLI Process

**What happens:** New code imports or calls submitted kernels from `src/sol_execbench/cli/main.py` or `src/sol_execbench/driver/problem_packager.py`.
**Why it's wrong:** It bypasses the subprocess isolation, stdout/stderr handling, reward-hack checks, and staged artifact contract used by `eval_driver.py`.
**Do this instead:** Stage source files through `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py` and execute `src/sol_execbench/driver/templates/eval_driver.py`.

### Changing Trace JSONL For Derived Reports

**What happens:** New diagnostic or summary fields are added directly to `Trace` for local reporting needs.
**Why it's wrong:** Trace JSONL is the canonical public output consumed by CLI users, baseline comparison, docs, and tests.
**Do this instead:** Put derived summaries in `src/sol_execbench/core/reporting.py` or diagnostic helpers in `src/sol_execbench/core/diagnostics.py`.

### Adding CUDA/NVIDIA Language Paths

**What happens:** New code accepts `cuda_cpp`, `cutlass`, `cudnn`, or `cuda_cflags` values.
**Why it's wrong:** This ROCm port intentionally rejects legacy CUDA schema values and maps native support to HIP/ROCm language categories.
**Do this instead:** Extend `SupportedLanguages` in `src/sol_execbench/core/data/solution.py` with ROCm-native categories and update `ProblemPackager._CPP_LANGUAGES` plus `eval_driver.py` native language sets.

### Timing Without Shifted Arguments

**What happens:** New benchmark timing calls invoke `torch.cuda.Event` directly around repeated calls with the same tensors.
**Why it's wrong:** It skips cache/address-shift semantics and can change measured behavior.
**Do this instead:** Use `time_runnable()` in `src/sol_execbench/core/bench/timing.py`, which uses `ShiftingMemoryPoolAllocator` from `src/sol_execbench/core/bench/io.py`.

## Error Handling

**Strategy:** Validate early with Pydantic models, isolate runtime failures into per-workload `Trace.evaluation.status`, and use subprocess exit codes only for unrecoverable compile/evaluation failures.

**Patterns:**
- Schema violations raise Pydantic validation errors when loading models in `src/sol_execbench/cli/main.py` and `src/sol_execbench/driver/templates/eval_driver.py`.
- CLI input problems raise `click.ClickException` in `src/sol_execbench/cli/main.py`.
- Compile subprocess failures are printed and exit with code 1 in `src/sol_execbench/cli/main.py:320`.
- Evaluation runtime failures usually emit `Trace` rows with statuses such as `INVALID_REFERENCE`, `RUNTIME_ERROR`, `INCORRECT_SHAPE`, `INCORRECT_DTYPE`, `INCORRECT_NUMERICAL`, or `REWARD_HACK`.
- `make_eval()` in `src/sol_execbench/core/bench/utils.py` attaches environment snapshots, timestamps, bounded logs, and optional extra messages.

## Cross-Cutting Concerns

**Logging:** CLI user-facing output uses Rich in `src/sol_execbench/cli/main.py`. Evaluation template redirects library stdout to stderr and writes strict JSON traces to the original stdout in `src/sol_execbench/driver/templates/eval_driver.py:261`. Clock-lock helpers use Python logging in `src/sol_execbench/core/bench/clock_lock.py`.

**Validation:** Pydantic models validate public schemas in `src/sol_execbench/core/data/`. AST parsing validates reference `run()` and input ordering in `src/sol_execbench/core/data/definition.py`. Tests under `tests/sol_execbench/test_public_contract_guardrails.py` and ROCm audit tests guard public behavior and migration constraints.

**Authentication:** Not applicable. The package evaluates local files and does not implement user identity or network authentication.

**Security:** Submitted source paths are constrained in `src/sol_execbench/core/data/solution.py`; Python solution attempts to call `torch.utils.cpp_extension.load/load_inline` are blocked inside `eval_driver.py`; reward-hack checks detect patched timing/eval functions, lazy outputs, and thread injection in `src/sol_execbench/core/bench/reward_hack.py`.

---

*Architecture analysis: 2026-05-22*
