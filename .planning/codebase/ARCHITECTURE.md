---
last_mapped: 2026-05-20
last_mapped_commit: unknown
focus: arch
---

# Architecture

## System Shape

SOL ExecBench is a local benchmarking framework. The public surface is a CLI
that loads a problem definition, workloads, and a solution, stages them into a
temporary directory, optionally compiles native code, then runs an isolated eval
driver subprocess that emits trace JSONL.

## Main Layers

1. CLI: `src/sol_execbench/cli/main.py`
   - Parses problem paths and options.
   - Loads `Definition`, `Workload`, `Solution`, and `BenchmarkConfig`.
   - Runs compile/evaluate subprocesses and renders Rich tables.

2. Driver: `src/sol_execbench/driver/problem_packager.py`
   - Writes validated problem files into staging.
   - Copies source files.
   - Injects GPU architecture flags for C++/CUDA solutions.
   - Provides compile and execute commands.

3. Eval templates: `src/sol_execbench/driver/templates/`
   - `build_ext.py` compiles native sources into `benchmark_kernel.so`.
   - `eval_driver.py` is the self-contained runtime script that imports user
     code, generates inputs, evaluates correctness, times performance, and
     emits `Trace` objects.

4. Core schemas: `src/sol_execbench/core/data/`
   - Pydantic models validate definitions, workloads, solutions, and traces.

5. Benchmark helpers: `src/sol_execbench/core/bench/`
   - Input generation, output normalization, correctness, timing, clock locking,
     environment snapshots, and reward-hack detection.

## Data Flow

`sol-execbench` loads JSON inputs -> constructs Pydantic models -> `ProblemPackager`
writes staging files -> C++ solutions compile to `benchmark_kernel.so` when
needed -> `eval_driver.py` executes workloads -> JSON trace lines return on
stdout -> CLI parses traces and exits nonzero if any workload fails.

## Execution Isolation

Evaluation runs in a subprocess instead of directly inside the CLI process.
`eval_driver.py` redirects library noise to stderr, preserves real stdout for
JSON, snapshots critical functions before importing user code, and validates
reward-hack conditions after import and calls.

## Extension Points

New solution languages generally require changes to `SupportedLanguages` in
`src/sol_execbench/core/data/solution.py`, C++ language detection in
`ProblemPackager`, and import/build handling in `eval_driver.py` or
`build_ext.py`.

New input formats should extend `src/sol_execbench/core/data/workload.py` and
`src/sol_execbench/core/bench/io.py`.
