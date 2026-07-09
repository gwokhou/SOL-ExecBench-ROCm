---
generated_by: gsd-map-codebase
generated_on: 2026-07-09
last_mapped_commit: cc007cd3af3e5100f7d86f155a40d5e51ffb57e5
focus: arch
---

# Architecture

## System Shape

SOL ExecBench ROCm Port is a layered local CLI evaluator. The CLI process loads
typed benchmark inputs, stages them into a temporary execution directory,
optionally compiles native HIP/C++ code, then runs a generated evaluator script
as a subprocess. The subprocess emits canonical trace JSONL and optional
diagnostic sidecars are written by the CLI.

The package boundary is `src/sol_execbench/`. Runtime code is separated into:

- `src/sol_execbench/cli/` for command parsing and user-facing workflows.
- `src/sol_execbench/core/` for schemas, benchmark logic, platform checks,
  dataset tooling, evidence, scoring, and reports.
- `src/sol_execbench/driver/` for staging, generated driver templates, and trace parsing.
- `src/sol_execbench/data/` for packaged static AMD hardware model data.

## Main Entry Points

- `sol-execbench` from `pyproject.toml`, implemented by `src/sol_execbench/cli/main.py`.
- `sol-execbench-baseline`, implemented by `src/sol_execbench/cli/commands/baseline.py`.
- Dataset operator script: `scripts/run_dataset.py`.
- Dataset download/migration helper: `scripts/download_solexecbench.py`.
- Docker wrapper: `scripts/run_docker.sh`.

`SolExecbenchCli` in `src/sol_execbench/cli/main.py` first offers metadata and
utility subcommands through `src/sol_execbench/cli/commands/`; if no subcommand
matches, it delegates to the evaluator command.

## Evaluation Data Flow

1. `src/sol_execbench/cli/main.py` parses CLI arguments.
2. `run_evaluation_cli()` in `src/sol_execbench/cli/evaluation/evaluator.py`
   resolves paths through `src/sol_execbench/cli/evaluation/problem_io.py`.
3. Pydantic models load `Definition`, `Workload`, `Solution`, and
   `BenchmarkConfig`.
4. `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py` creates a
   `tempfile.mkdtemp(prefix="sol_execbench_")` staging directory and writes
   normalized `definition.json`, `workload.jsonl`, `solution.json`,
   `config.json`, solution sources, and stageable safetensors inputs.
5. Native ROCm solutions invoke `ProblemPackager.compile()`, which writes
   `src/sol_execbench/driver/templates/build_ext.py`, injects HIP offload
   architecture flags when needed, and builds `benchmark_kernel.so`.
6. `ProblemPackager.execute()` writes
   `src/sol_execbench/driver/templates/eval_driver.py` and returns the command
   to run in staging.
7. `src/sol_execbench/cli/evaluation/runtime.py` executes the driver, optionally
   via `rocprofv3`, and parses stdout into `Trace` models.
8. `src/sol_execbench/cli/evaluation/outputs.py` writes trace JSONL and prints
   output; `src/sol_execbench/cli/evaluation/sidecar_writer.py` writes optional
   profile, static evidence, profile-summary, agent-feedback, and environment
   sidecars.

## Subprocess Boundary

Submitted solution code is not imported into the CLI process. It is imported by
the staged `eval_driver.py` subprocess. This boundary keeps user code, native
extension build products, profiler artifacts, generated runtime files, and
reference code out of the long-lived CLI process.

The boundary is an execution guardrail, not a sandbox. Documentation in
`README.md`, `docs/ARCHITECTURE.md`, and `docs/SECURITY.md` says untrusted
submissions require external isolation such as Docker, a VM, or a dedicated
ROCm host.

## Core Abstractions

- `Definition` in `src/sol_execbench/core/data/definition.py` describes inputs,
  outputs, symbolic axes, and reference source.
- `Workload` in `src/sol_execbench/core/data/workload.py` represents generated,
  scalar, custom, or safetensors workload rows and tolerances.
- `Solution` in `src/sol_execbench/core/data/solution_instance.py` captures
  source files, entry point, target hardware, dependencies, and content hash.
- `BuildSpec` and `CompileOptions` in
  `src/sol_execbench/core/data/solution_models.py` enforce ROCm category and
  compile-option constraints.
- `BenchmarkConfig` in `src/sol_execbench/core/bench/config/benchmark_config.py`
  controls warmup, iteration, timing, seed, and clock-lock behavior.
- `Trace` in `src/sol_execbench/core/data/trace.py` is the canonical per-workload result.
- `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py` owns staging,
  native compile command generation, and evaluation driver command generation.

## Benchmark Runtime

Staged `eval_driver.py` imports helpers through
`src/sol_execbench/driver/eval_runtime_api.py`, which exports benchmark runtime
functions from `src/sol_execbench/core/bench/` and schema models from
`src/sol_execbench/core/data/`.

Benchmark helpers cover:

- Input generation and heuristics in `input_generation.py` and `input_heuristics.py`.
- Output allocation and destination-passing support in `output_allocation.py`.
- Correctness checks in `correctness.py`, `eval_correctness.py`, and
  `eval_output_integrity.py`.
- Timing in `timing.py`, `eval_timing.py`, `timing_policy.py`, and
  `timing_isolation.py`.
- Reward-hack checks in `reward_hack/`.
- Memory-pool behavior in `memory_pool.py`.
- PID locking and GPU process isolation in `pid_lock.py` and `timing_isolation.py`.

## Evidence And Reporting Layers

Diagnostic evidence is intentionally not part of correctness authority:

- `src/sol_execbench/core/bench/rocm_profiler/` handles optional `rocprofv3`
  profile and timing artifacts.
- `src/sol_execbench/core/bench/static_kernel/` handles static kernel sidecars.
- `src/sol_execbench/core/bench/profile_summary/` normalizes profile summaries.
- `src/sol_execbench/core/bench/agent_feedback/` builds bounded agent feedback sidecars.
- `src/sol_execbench/core/evidence/` stores evidence refs and runtime evidence collectors.

Reports in `src/sol_execbench/core/reports/` and scoring in
`src/sol_execbench/core/scoring/` consume traces and sidecars to produce bounded
analysis artifacts.

## Dataset Layer

`src/sol_execbench/core/dataset/` is the package-owned dataset workflow layer.
It covers local migration, inventory, readiness, execution closure, run-state
reuse, deterministic sharding, paper denominator reports, parity gaps, profiler
timing coverage, and release-oriented validation. `scripts/run_dataset.py`
is the operator-facing script that composes these helpers.

## Platform Layer

`src/sol_execbench/core/platform/` owns ROCm environment diagnostics, dependency
matrix models, Docker target manifests, compatibility matrix reports, and
toolchain routing. CLI commands under `src/sol_execbench/cli/commands/` expose
the most important platform metadata through `doctor`, `toolchain`, and
`contract` style subcommands.
