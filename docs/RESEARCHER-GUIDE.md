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
| benchmark/reproducibility researcher | Use the curated slice, environment sidecars, profiling sidecars, static evidence sidecars, execution closure, v1.19 sidecar reports, and v1.20 evidence-quality reports. | `docs/curated_rocm_slice.md`, `docs/static_kernel_evidence.md`, `docs/v1_19_evidence_guide.md`, `docs/v1_20_evidence_quality_guide.md`, `docs/CLAIMS.md`, execution closure, paper denominator gaps, Matrix diagnostics, AMD bound sanity, consistency, stability, claim-upgrade, and trust summary. |

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

For v1.19 evidence surfaces, start with `docs/v1_19_evidence_guide.md`. It
collects the CPU-safe command shapes and interpretation rules for execution
closure, paper denominator reports, Matrix schema export, Matrix semantic diff,
and AMD bound sanity. These reports have no full 235-problem paper validation,
no upstream SOLAR parity, no score authority, no leaderboard readiness, no CDNA
3/MI300X/CDNA4 validation, no native-host ROCm Matrix validation, and no
new-hardware validation.

For v1.20 evidence-quality review, continue with
`docs/v1_20_evidence_quality_guide.md`. It covers consistency lint, evaluation
stability, claim-upgrade rules, and trust summaries. These reports are
sidecar-only and diagnostic; they do not create correctness, timing, score,
paper-parity, native-host, new-hardware, or leaderboard authority.

## Interpreting Artifacts

| Artifact | What it tells you | What it does not prove |
| --- | --- | --- |
| Trace JSONL | Workload correctness, measured latency, reference latency, environment fields. | Paper parity or hardware roofline validity. |
| Environment sidecar | ROCm tools, device identity, PyTorch ROCm readiness, event timing readiness. | Correctness or score authority. |
| Profile sidecar | `rocprofv3` command provenance, artifacts, status, stdout/stderr tails. | Correctness or SOL score authority. |
| Static evidence sidecar | Current-build HIP/C++ artifacts, hashes, routed `llvm-objdump` / `readelf` records, bounded raw output paths, and diagnostic status. | Correctness, timing, score, paper parity, leaderboard readiness, CDNA 3/CDNA 4 validation, Triton cache coverage, RGA-rich resource parsing, or paper-scale static coverage. |
| AMD SOL sidecar | Derived AMD bound graph, estimates, hardware model, and coverage state. | Upstream SOLAR equivalence. |
| AMD score report | Guarded local AMD-native score interpretation. | NVIDIA B200 or leaderboard equivalence. |
| Execution closure | Which scoped problems have closure statuses: `not_attempted`, `filtered`, `skipped_existing_pass`, `attempted_passed`, `attempted_failed`, `missing_trace`, or `derived_evidence_missing`. | Full 235-problem validation unless the denominator is actually complete. |
| Paper denominator report | Bounded denominator accounting, source refs, checksums, evidence gaps, and false claim-boundary fields. | Paper parity, score authority, or leaderboard readiness. |
| Matrix schema export | Strict Matrix JSON Schema shape for report validation. | Native-host ROCm Matrix validation or hardware validation. |
| Matrix semantic diff | Diagnostic Matrix report drift, severity, and review context. | Score authority, leaderboard readiness, or clean hardware validation. |
| AMD bound sanity report | Existing-evidence source refs, checksums, warnings, and bounded AMD/SOL/SOLAR consistency checks. | Upstream SOLAR parity, new-hardware validation, or score authority. |

## Adding Or Adapting A Kernel

1. Start from the nearest existing example under `examples/` or
   `tests/sol_execbench/samples/`.
2. Keep the canonical problem files unchanged unless you are intentionally
   defining a new benchmark problem.
3. Update or add a solution JSON file that references your implementation.
4. Run the single-problem CLI and inspect correctness before interpreting
   latency.
5. If the solution is native HIP/C++, use explicit AMD targets or `LOCAL`.
6. Use `--static-evidence auto` when you need diagnostic build artifacts and
   routed static extractor sidecars.
7. Do not bypass reward-hack checks by loading hidden native extensions,
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
- Mark unavailable, unsupported, partial, failed, or skipped static evidence
  explicitly.
- Report denominator counts for any dataset or curated-slice result.
- For v1.19 sidecars, keep source refs relative, checksums visible, logs
  bounded, and authority fields false.
- Link claims to `docs/CLAIMS.md`.
