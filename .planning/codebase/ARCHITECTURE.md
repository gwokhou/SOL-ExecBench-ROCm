---
generated_by: gsd-map-codebase
focus: arch
mapped_at: 2026-06-16
---

# Architecture

## System Shape

SOL ExecBench ROCm Port is a local Python CLI package for evaluating benchmark
problems and solution manifests on AMD ROCm hardware. The public interface is a
Click CLI. The internal shape is layered:

- `src/sol_execbench/cli/` handles commands, file loading, subprocess execution,
  diagnostics sidecars, and user-facing output.
- `src/sol_execbench/core/data/` defines Pydantic schemas for definitions,
  workloads, solutions, traces, shapes, dtypes, and the evaluator contract.
- `src/sol_execbench/driver/` stages problems and writes generated build/eval
  driver templates into temporary directories.
- `src/sol_execbench/core/bench/` contains correctness, timing, input,
  reward-hack, profiling, static-evidence, and runtime helpers.
- `src/sol_execbench/core/dataset/` contains dataset discovery, migration,
  readiness, denominator, closure, sharding, and runner support.
- `src/sol_execbench/core/scoring/` contains AMD bound, AMD score, SOLAR
  derivation, baseline artifact, and hardware-model helpers.
- `scripts/` contains operator workflows layered on top of importable package
  code.

## Entry Points

- `sol-execbench` dispatches to `sol_execbench.cli:cli`.
- `sol-execbench-baseline` dispatches to `sol_execbench.cli.baseline:cli`.
- Metadata subcommands are implemented in `src/sol_execbench/cli/main.py`:
  `contract`, `doctor`, `toolchain`, and `dataset`.
- Dataset-scale runner entry point is `scripts/run_dataset.py`.
- Docker wrapper entry point is `scripts/run_docker.sh`.

## Evaluation Data Flow

1. `src/sol_execbench/cli/main.py` resolves a problem directory or explicit
   `--definition`, `--workload`, `--solution`, and optional `--config` files.
2. Inputs are loaded into `Definition`, `Workload`, `Solution`, and
   `BenchmarkConfig` objects.
3. `ProblemPackager` in `src/sol_execbench/driver/problem_packager.py` writes
   normalized files into a temporary staging directory.
4. Source files from `Solution.sources` are copied into staging.
5. Safetensors inputs may be symlinked or copied into staging when the repo or
   `FLASHINFER_TRACE_DIR` contains local blobs.
6. Native ROCm solutions compile through generated `build_ext.py`.
7. Evaluation runs through generated `eval_driver.py` in a subprocess.
8. The driver emits one strict JSON `Trace` per workload to stdout.
9. The CLI parses traces, writes optional output/sidecars, prints Rich tables,
   and returns nonzero when any workload fails.

## Subprocess Boundary

- User solution code is staged and run outside the CLI process.
- The generated eval driver redirects noisy stdout to stderr before importing
  PyTorch/Triton, then writes canonical JSONL to the saved real stdout.
- Reward-hack checks in `src/sol_execbench/core/bench/reward_hack.py` perform
  static source review and runtime integrity checks.
- This boundary is an execution guardrail, not a hardened sandbox.
- Untrusted submissions still require Docker, VM, or host isolation.

## Dataset And Reporting Flow

- `scripts/run_dataset.py` owns argument parsing, output layout, execution
  phases, and operator-facing orchestration.
- Reusable dataset helper logic lives in `src/sol_execbench/core/dataset/`.
- Reports are built from trace JSONL, closure records, sidecars, matrix data,
  and scoring artifacts.
- Derived report modules include `src/sol_execbench/core/claim_upgrade.py`,
  `src/sol_execbench/core/consistency.py`,
  `src/sol_execbench/core/evaluation_stability.py`,
  `src/sol_execbench/core/matrix_diff.py`,
  `src/sol_execbench/core/runtime_evidence.py`, and
  `src/sol_execbench/core/trust_summary.py`.

## Extension Points

- Add or modify public schema fields in `src/sol_execbench/core/data/` with
  matching tests and docs.
- Add solution categories in `SupportedLanguages` and the native/Python routing
  logic in `src/sol_execbench/core/data/solution.py`,
  `src/sol_execbench/driver/problem_packager.py`, and driver templates.
- Add ROCm evidence tools through `src/sol_execbench/core/toolchain.py`,
  `src/sol_execbench/core/bench/rocm_profiler.py`, or
  `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- Add dataset phases through `src/sol_execbench/core/dataset/` helpers and
  `scripts/run_dataset.py`.

## Important Boundaries

- Public schemas must remain stable unless a change is intentional and tested.
- Diagnostic sidecars do not replace canonical trace JSONL.
- ROCm validation claims are intentionally conservative and documented in
  `docs/CLAIMS.md`, `docs/rocm.md`, and related release docs.
- NVIDIA/CUDA legacy categories are rejected or treated as legacy examples in
  this ROCm-only port.
