<!-- generated-by: gsd-doc-writer -->
# Researcher Guide

This guide is for GPU kernel researchers and deep developers using the ROCm
port of SOL ExecBench. It explains where to start, which artifacts matter, and
how to avoid overstating results.

## Choose Your Path

Start from the question you are trying to answer:

| Question | Start with | Primary artifacts |
| --- | --- | --- |
| Did this kernel produce correct outputs and useful speedup on my AMD host? | Run one problem through `sol-execbench`, then inspect Trace JSONL. | Trace JSONL, correctness fields, latency, reference latency, environment fields. |
| How do I add or adapt a kernel implementation? | Read `docs/solution.md`, then copy the closest example under `examples/` or `tests/sol_execbench/samples/`. | Solution JSON, staged source files, compile options, HIP/Triton examples. |
| How does the harness stage and execute solutions? | Inspect `src/sol_execbench/driver/` and `src/sol_execbench/driver/templates/`. | Staging directory, generated driver, native build path, reward-hack traces. |
| How reproducible is this dataset or slice result? | Run a bounded dataset batch with readiness and execution closure sidecars. | Trace JSONL, environment sidecars, execution closure, readiness, trust summaries. |
| Can I review a public prerelease claim? | Start with `docs/research_preview.md`, then inspect artifact bundle and readiness outputs. | Prerelease bundle, readiness report, `docs/CLAIMS.md`, known-gap records. |

These questions map to the common roles from earlier docs: GPU kernel author,
compiler/backend researcher, agent kernel-optimization researcher,
benchmark/reproducibility researcher, and research preview reviewer.

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

For the v1.26 research preview, start with `docs/research_preview.md`. It maps
methodology, scope, evidence surfaces, limitations, and representative commands
to expected artifacts.

Use older evidence guides when you need their specific sidecars:

- `docs/static_kernel_evidence.md` covers diagnostic HIP/C++ build artifacts
  collected with `--static-evidence auto`.
- `docs/prerelease_artifact_bundle.md` covers the versioned public review
  bundle.
- `docs/prerelease_readiness.md` covers the readiness gate for missing
  evidence, checksum drift, known gaps, and claim-boundary regressions.
- `docs/v1_19_evidence_guide.md` covers execution closure, paper denominator
  reports, Matrix schema export, Matrix semantic diff, and AMD bound sanity.
- `docs/v1_20_evidence_quality_guide.md` covers consistency lint, evaluation
  stability, claim-upgrade rules, and trust summaries.

Those reports are sidecars and review aids. They do not replace canonical Trace
JSONL or upgrade the project to paper parity, upstream SOLAR parity, score
authority, leaderboard readiness, native-host validation, MI300X validation, or
CDNA4 validation.
For the older v1.19/v1.20 wording specifically, they provide no
CDNA3-family validation, including MI300X, and no CDNA4 validation.

## Interpreting Artifacts

| Artifact | What it tells you | What it does not prove |
| --- | --- | --- |
| Trace JSONL | Workload correctness, measured latency, reference latency, environment fields. | Paper parity or hardware roofline validity. |
| Environment sidecar | ROCm tools, device identity, PyTorch ROCm readiness, event timing readiness. | Correctness or score authority. |
| Profile sidecar | `rocprofv3` command provenance, artifacts, status, stdout/stderr tails. | Correctness or SOL score authority. |
| Static evidence sidecar | Current-build HIP/C++ artifacts, hashes, routed `llvm-objdump` / `readelf` records, bounded raw output paths, and diagnostic status. | Correctness, timing, score, paper parity, leaderboard readiness, CDNA 3/CDNA 4 validation, Triton cache coverage, RGA-rich resource parsing, or paper-scale static coverage. |
| AMD SOL sidecar | Derived AMD bound graph, estimates, hardware model, and coverage state. | Upstream SOLAR equivalence. |
| AMD score report | Guarded local AMD-native score interpretation. | NVIDIA B200 or leaderboard equivalence. |
| Execution closure | Which scoped problems have closure statuses: `not_attempted`, `filtered`, `skipped_existing_pass`, `attempted_passed`, `attempted_failed`, `missing_trace`, `derived_evidence_missing`, or `excluded_long_tail`. | Full 235-problem validation unless the denominator is actually complete. |
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
