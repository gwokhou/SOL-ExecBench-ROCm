# Researcher Guide

This guide is for GPU kernel researchers and deep developers using the ROCm
port of SOL ExecBench. It explains where to start, which artifacts matter, and
how to avoid overstating results.

## Choose Your Path

| Role | Start with | Primary artifacts |
| --- | --- | --- |
| GPU kernel author | Run one local example, then adapt a solution file. | `solution.json`, trace JSONL, correctness/performance fields. |
| compiler/backend researcher | Inspect solution schemas, staging, and native build paths. | `docs/solution.md`, `src/sol_execbench/driver/`, HIP/Triton examples. |
| agent kernel-optimization researcher | Use the isolated harness and reward-hack checks as the execution boundary. | trace JSONL, `REWARD_HACK` traces, baseline comparisons. |
| benchmark/reproducibility researcher | Use the curated slice, environment sidecars, profiling sidecars, and closure reports. | `docs/curated_rocm_slice.md`, `docs/CLAIMS.md`, execution closure, parity gaps. |

## First Run

```bash
uv sync --all-groups

uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
  --json \
  -o out/researcher/rmsnorm.trace.jsonl
```

Read the canonical trace first. It is the primary artifact. Derived reports and
sidecars must not change trace JSONL semantics.

## Interpreting Artifacts

| Artifact | What it tells you | What it does not prove |
| --- | --- | --- |
| Trace JSONL | Workload correctness, measured latency, reference latency, environment fields. | Paper parity or hardware roofline validity. |
| Environment sidecar | ROCm tools, device identity, PyTorch ROCm readiness, event timing readiness. | Correctness or score authority. |
| Profile sidecar | `rocprofv3` command provenance, artifacts, status, stdout/stderr tails. | Correctness or SOL score authority. |
| AMD SOL sidecar | Derived AMD bound graph, estimates, hardware model, and coverage state. | Upstream SOLAR equivalence. |
| AMD score report | Guarded local AMD-native score interpretation. | NVIDIA B200 or leaderboard equivalence. |
| Execution closure | Which scoped problems were attempted, skipped, blocked, passed, failed, or unscored. | Full 235-problem validation unless the denominator is actually complete. |

## Adding Or Adapting A Kernel

1. Start from the nearest existing example under `examples/` or
   `tests/sol_execbench/samples/`.
2. Keep the canonical problem files unchanged unless you are intentionally
   defining a new benchmark problem.
3. Update or add a solution JSON file that references your implementation.
4. Run the single-problem CLI and inspect correctness before interpreting
   latency.
5. If the solution is native HIP/C++, use explicit AMD targets or `LOCAL`.
6. Do not bypass reward-hack checks by loading hidden native extensions,
   external files, subprocesses, network calls, semantic caches, or hidden
   streams.

## Compiler And Backend Research

The most useful integration points are:

- `src/sol_execbench/core/data/solution.py` for solution metadata and supported
  language categories.
- `src/sol_execbench/driver/problem_packager.py` for staging and build command
  generation.
- `src/sol_execbench/driver/templates/` for generated evaluation and native
  extension build logic.
- `src/sol_execbench/core/bench/timing.py` for HIP-backed event timing.
- `src/sol_execbench/core/scoring/` for derived AMD score and bound artifacts.

Compiler experiments should report whether they changed solution source,
compile flags, runtime library paths, or scoring interpretation. Performance
claims need comparable hardware, ROCm version, clock policy, and workload
inputs.

## Agent Optimization Research

Agents should treat `sol-execbench` as the only execution authority:

- Generate candidate solution files.
- Run the CLI or dataset runner.
- Read trace JSONL and explicit sidecars.
- Avoid hidden state between attempts unless it is represented in the solution
  source and allowed by the schema.
- Use baseline comparison for candidate-vs-candidate analysis.
- Use AMD-native score reports only when required bound and baseline evidence
  exists.

Never train or route an agent on private derived artifacts that are not
available to the evaluation policy.

## Reproducibility Checklist

- Record git commit and command line.
- Record ROCm version, PyTorch ROCm version, GPU architecture, and clock policy.
- Keep canonical traces separate from sidecars.
- Mark unavailable profiler, hardware, library, or score evidence explicitly.
- Report denominator counts for any dataset or curated-slice result.
- Link claims to `docs/CLAIMS.md`.
