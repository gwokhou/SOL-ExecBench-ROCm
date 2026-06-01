# Analysis And Profiling

SOL ExecBench emits one JSON trace per workload. Use these traces as the
primary analysis artifact, then use ROCm profiling tools for deeper kernel
inspection when needed.

## Trace Collection

Write JSONL traces from a single problem:

```bash
uv run sol-execbench examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json \
  --json \
  -o out/rmsnorm.jsonl
```

Run a dataset batch:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5
```

Before running dataset batches, the local public dataset layout can be checked
with a sidecar acquisition/layout manifest:

```bash
python scripts/download_solexecbench.py \
  --verify-only \
  --output-root data/SOL-ExecBench/benchmark \
  --manifest out/dataset_manifest.json
```

The manifest records dataset source, selected categories, shallow counts,
checksums, diagnostics, and claim-boundary fields. Acquisition/layout completion
does not prove ROCm readiness, execution success, paper-level validation, hosted
leaderboard parity, or upstream SOLAR equivalence.

Generate static paper-inventory and ROCm-readiness sidecars:

```bash
uv run scripts/inspect_dataset.py \
  --dataset-root data/SOL-ExecBench/benchmark \
  --manifest out/dataset_manifest.json \
  --inventory out/inventory.json \
  --readiness out/readiness.json \
  --ready-subset out/ready_subset.json
```

Readiness means a workload is ready to attempt local ROCm execution. It does
not mean the workload passed execution, was scored, completed full validation,
or reached paper parity.

Run a bounded ready subset and write an execution-closure sidecar:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --ready-subset out/ready_subset.json \
  --readiness out/readiness.json \
  --dataset-manifest out/dataset_manifest.json \
  --limit 5 \
  --max-workloads 1 \
  --execution-closure out/execution_closure.json
```

When `--ready-subset` is supplied, the runner filters execution to the ready
problem/workload refs, still invokes the primary `sol-execbench` subprocess
path, and writes temporary filtered workload files only under the selected
output directory. The execution-closure sidecar joins ready-subset selection,
optional readiness blockers, traces, summaries, skipped-existing-pass states,
missing traces, failures, and derived evidence refs. The closure sidecar is a
bounded local execution audit. It is not full 235-problem validation,
not paper parity, and not a leaderboard result.

Generate a parity-gap JSON and Markdown report from the sidecars:

```bash
uv run scripts/report_parity_gaps.py \
  --manifest out/dataset_manifest.json \
  --inventory out/inventory.json \
  --readiness out/readiness.json \
  --ready-subset out/ready_subset.json \
  --execution-closure out/execution_closure.json \
  --amd-score-report out/amd-score-report.json \
  --json-output out/parity_gap_report.json \
  --markdown-output out/parity_gap_report.md
```

The parity-gap report summarizes discovered, parsed, ready, blocked,
not-attempted, skipped, attempted, passed, failed, scored, degraded, and
unscored denominators. It groups blockers by reason code with next actions and
separates trace, timing, AMD-native score, AMD SOL, and SOLAR derivation
evidence completeness. This report is a bounded local gap report,
not full validation, not paper parity, not upstream SOLAR parity,
not NVIDIA B200 or Blackwell equivalence, and not hosted leaderboard readiness.

Dataset runs write per-problem traces under category/problem subdirectories of
the selected output directory and write a run summary sidecar directly under
that output directory. The default output directory is `out/`.

For source-specific ROCm timing evidence, add a timing evidence directory and
record the target architecture:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --limit 5 \
  --timing-evidence-dir out/timing-evidence \
  --gpu-architecture gfx942 \
  --timing-tool-version "rocprofv3 7.0.0"
```

HIP-native and Triton sources use `rocprofv3` kernel activity timing when the
tool is available. PyTorch/operator and unsupported mixed-source cases emit an
explicit fallback selection instead of being labeled as kernel activity timing.

## Timing Method

The ROCm port does not use CUPTI. Timing uses PyTorch's HIP-backed device event
API through the historical `torch.cuda.Event` namespace. This is intentional:
PyTorch exposes ROCm GPU devices through `torch.cuda` compatibility APIs.

The benchmark path:

1. pre-allocates distinct input/output buffers for timed iterations,
2. clears a GPU cache-sized tensor before each iteration,
3. records HIP-backed device events around the solution call,
4. synchronizes before reading elapsed time,
5. emits median latency by default.

## Clock Stability

For benchmark-grade runs, use:

```bash
uv run sol-execbench <problem_dir> --solution <solution-file> --lock-clocks
```

Clock locking uses `rocm-smi`. The command fails the workload if
`--lock-clocks` is requested but the environment did not lock clocks before the
evaluation subprocess starts. This prevents silently mixing locked and unlocked
timing data.

## Reward-Hack Review

Before user code is imported, the evaluation driver statically reviews submitted
source text for exploit patterns that can distort correctness or timing. Blocked
patterns include non-default stream creation, data-pointer or content-keyed
semantic caches, unauthorized file I/O, embedded payload decoding, dynamic native
loading, subprocess or shell process execution, network access, and precision
downgrades for float32 output contracts.

The review emits `REWARD_HACK` traces with rule names such as
`hidden_async_stream`, `semantic_output_cache`,
`unauthorized_file_or_loader`, and `precision_downgrade`. Existing runtime
guards still check timing monkey-patches, eval-driver integrity, thread
injection, and lazy/proxy outputs.

## External ROCm Profiling

Use `rocprofv3` around a small, representative command when kernel-level
analysis is needed:

```bash
rocprofv3 --stats -- \
  uv run sol-execbench examples/pytorch/gemma3_swiglu \
    --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

For HIP/C++ solutions, keep compilation outside the profiling window when
possible by running once with `--keep-staging`, then profiling a repeated run.
This keeps compile overhead out of kernel timing analysis.

Dataset timing evidence JSON records:

- source type, selected backend, activity domain, aggregation rule, and fallback
  reason when applicable,
- `rocprofv3` command, return code, stdout/stderr, and parsed CSV path,
- tool version, GPU architecture, warmup runs, measured iterations, trial count,
  and clock-lock status,
- parsed profiler rows and aggregate kernel duration for profiler-backed runs.

## Interpreting Results

Key trace fields:

- `evaluation.status`: final outcome for the workload.
- `evaluation.correctness`: maximum absolute/relative error and non-finite
  flags.
- `evaluation.performance.latency_ms`: measured solution latency.
- `evaluation.performance.reference_latency_ms`: PyTorch reference latency on
  the same hardware.
- `evaluation.environment`: AMD hardware and ROCm/PyTorch library versions.

Do not compare latencies across machines unless ROCm version, GPU architecture,
clock policy, and problem inputs are comparable.

## Baseline Comparison

Use `sol-execbench-baseline` to compare existing trace JSONL files without
changing the trace schema:

```bash
uv run sol-execbench-baseline \
  --candidate out/candidate.jsonl \
  --baseline out/baseline.jsonl
```

The comparison matches traces by `definition` and `workload.uuid`, uses the
fastest passed baseline for each workload, and classifies each candidate result
as:

- `WIN`: candidate beats the baseline by at least `--win-pct` percent.
- `PARITY`: candidate is within `--parity-pct` percent of the baseline.
- `LOSS`: candidate is slower than the parity threshold.
- `NO_BASELINE` or `NO_CANDIDATE`: one side lacks a passed trace with
  performance data.

JSON output is available for automation:

```bash
uv run sol-execbench-baseline \
  --candidate out/candidate.jsonl \
  --baseline out/baseline.jsonl \
  --format json \
  --output out/baseline-comparison.json
```

The `out/` path above is an example output location created by the command.

Baseline comparison is baseline-relative. It is not an AMD-native roofline claim
unless you provide and validate a separate AMD interpretation model. The
`--amd-native-claim` flag intentionally emits a warning so reports cannot
silently present benchmark-relative data as hardware validation.

## AMD-Native Score Interpretation

The original SOL-Score formula is preserved for compatibility, but this ROCm
port is not an NVIDIA/B200/SOLAR equivalence study. In this ROCm port:

1. `sol_score()` remains a formula helper.
2. Baseline comparison reports are baseline-relative by default.
3. AMD-native claims require an explicit AMD roofline or equivalent model,
   recorded hardware evidence, and documentation of which architecture and
   clock policy the model covers.
4. AMD-native scoring requires an AMD SOL bound artifact with graph nodes,
   FLOP/byte evidence, hardware model source, confidence, and split validation
   states (`hardware_validation_status`, `model_validation_status`)
   before reporting AMD-native scores.
5. CDNA 3 claims additionally require real `gfx94*` full-suite validation
   evidence, which is not part of the v1.9 milestone.

AMD-native score reports are derived artifacts. They can reference trace timing,
ROCm timing evidence, baseline summaries, and AMD SOL bound artifacts, but they
do not add fields to canonical trace JSONL. These reports are AMD ROCm
interpretation artifacts: not NVIDIA B200, SOLAR, or leaderboard equivalence claims.

Dataset runs can optionally write a derived AMD-native suite score report:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --limit 5 \
  --amd-score-report out/amd-score-report.json
```

The AMD score report path is caller supplied; the path shown in the command
above is only an example output location.

The report is opt-in. It reads canonical trace output and derived AMD SOL bound
inputs, records trace, timing, SOL-bound, baseline, and hardware-model evidence
references for each workload score, and keeps the output separate from canonical
trace JSONL. Missing timing, baseline, or bound evidence is reported as an
unscored guarded state rather than an invented score.

To also materialize the derived AMD SOL bound artifact v2 sidecars used by the
score report, provide a sidecar directory:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --limit 5 \
  --amd-score-report out/amd-score-report.json \
  --amd-sol-bound-dir out/amd-sol-bounds
```

The sidecars use schema version `sol_execbench.amd_sol_bound.v2`. Each sidecar
contains the derived marker, definition name, workload UUID, hardware model
reference, hardware model payload, bound graph, rich operator work estimates,
per-operation SOL bounds, aggregate bound state, deterministic warnings, and
coverage summary. This is the ROCm port's AMD-local analog of the paper's
graph/evidence/SOL-analyzer artifact boundary; it is not an upstream NVIDIA
B200 or full SOLAR reproduction claim.

For v1.10 SOLAR derivation evidence, add `--solar-derivation` alongside the
AMD-native report and AMD SOL bound sidecar options:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --limit 5 \
  --amd-score-report out/amd-score-report.json \
  --amd-sol-bound-dir out/amd-sol-bounds \
  --solar-derivation out/solar-derivation
```

This local workflow produces three artifact layers:

- canonical trace JSONL files: the benchmark trace JSONL output for each problem.
  The canonical trace JSONL remains unchanged by `--amd-score-report`,
  `--amd-sol-bound-dir`, and `--solar-derivation`.
- AMD SOL v2 sidecars under `--amd-sol-bound-dir`: derived AMD roofline and
  work-estimate inputs used for AMD-native score eligibility.
- SOLAR derivation sidecars under `--solar-derivation`: paper-aligned automatic
  SOLAR derivation evidence for the ROCm port, generated from `Definition`,
  `Workload`, reference-visible source, and other static evidence. The
  derivation boundary does not use candidate solution execution or benchmark
  timing to create formulas.
- AMD-native score reports at `--amd-score-report`: opt-in local derived
  reports that consume canonical traces plus derived sidecars and expose
  workload score state, warnings, and audit references.

Inspect `coverage_summary` to understand how many operation groups are
supported, inexact, or unsupported. Inspect `aggregate_status` before treating
any score as usable: `scored` is eligible for AMD-native score output,
`degraded` is an auditable provisional state with warnings, and `unscored`
means score eligibility failed and the AMD-native score is intentionally
omitted. Report warnings explain the guard that produced the state. Report
entries also carry `derived_evidence_refs` for formula evidence, hardware model
evidence, coverage, and score eligibility; these refs are derived-report-only
metadata and do not change public trace fields or the canonical `evidence_refs`
shape.

For release-defined scoring, provide an optimized scoring baseline artifact:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --limit 5 \
  --amd-score-report out/amd-score-report.json \
  --scoring-baseline <path-to-scoring-baseline.json>
```

Baseline artifacts are derived JSON inputs keyed by definition and workload UUID:

```json
{
  "schema_version": "sol_execbench.scoring_baseline.v1",
  "release": "v1.7",
  "entries": [
    {
      "definition": "example_problem",
      "workload_uuid": "workload-uuid",
      "latency_ms": 0.123,
      "solution": "optimized_baseline"
    }
  ]
}
```

When no matching scoring baseline artifact entry exists, AMD-native reports may
fall back to `trace.evaluation.performance.reference_latency_ms`, but the score
is labeled with `baseline_source: reference_latency` and carries a provisional
baseline warning. Treat `baseline_source: scoring_baseline` as the release-style
path; treat `reference_latency` as a development fallback.

## AMD SOL Bound Artifact V2 Semantics

The v2 AMD SOL bound artifact is a derived sidecar. It does not modify
`Definition`, `Workload`, `Trace`, `Solution`, or primary `sol-execbench` CLI
schemas. The artifact exists to make AMD-native score inputs auditable:

- `bound_graph`: workload-bound operation and tensor evidence.
- `operator_work_estimates`: per-operation formulas, FLOPs, byte buckets,
  movement bytes, confidence, rationale, and warnings.
- `op_bounds`: per-operation compute bound, memory bound, SOL bound, limiting
  resource, confidence, and rationale.
- `aggregate_bound`: the score-eligibility state for the workload.
- `coverage_summary`: supported, inexact, and unsupported counts by operation
  family plus worst confidence.
- `warnings`: deterministic degradation categories for callers and reports.

Aggregate status is conservative:

- `scored`: all required operation and hardware evidence is supported and
  validated.
- `degraded`: evidence is usable for a provisional derived score but contains
  inexact estimates or provisional hardware/model validation.
- `unscored`: unsupported or missing bound evidence is present, so the
  AMD-native workload score is omitted instead of treating missing work as
  zero-cost.

Hardware model payloads carry both `hardware_validation_status` and
`model_validation_status`. For v1.9, RDNA 4 (`gfx1200`) is the only validation
target. CDNA 3 / MI300X real-hardware validation and CDNA 4 validation remain
future work; reports must not present those paths as validated in this
milestone.

## v1.10 Claim Boundaries

v1.10 derived sidecars and reports support the local
`claim_level: amd-native-derived` interpretation. They provide paper-aligned
automatic SOLAR derivation evidence for this ROCm port, but they are not
paper-scale 124-model / 235-problem extraction, not upstream SOLAR parity, not
NVIDIA B200 or Blackwell equivalence, not hosted leaderboard readiness, and not
new real-hardware validation. They also do not claim CDNA 3 / MI300X
validation, CDNA 4 validation, NVFP4 validation, MXFP4 validation, or any new
hardware validation beyond the evidence explicitly recorded in the local
artifacts.

Use these artifacts as auditable AMD ROCm derived evidence. Do not present
`coverage_summary`, `aggregate_status`, `derived_evidence_refs`, or score
eligibility as proof of original-paper parity, upstream SOLAR parity, Blackwell
or B200 equivalence, leaderboard readiness, or benchmark-scale hardware
validation.

## AMD SOL Coverage Semantics

AMD SOL bound artifacts include a derived coverage summary before scores are
reported. Coverage labels distinguish supported, inexact, and unsupported
operation evidence:

- `supported`: an analyzer has a direct, auditable estimate for the operation.
- `inexact`: the operation is recognized, but the FLOP or byte estimate is
  conservative and should not be presented as exact hardware validation.
- `unsupported`: the operation remains visible in the artifact and must not be
  silently treated as complete SOL evidence.

Shape, view, transpose, broadcast, and reshape-like operations are modeled as
data-movement or zero-FLOP nodes when recognized. Reductions, normalization,
softmax-like operations, and activations use conservative estimates unless a
future analyzer records a more exact formula. These coverage summaries are
derived methodology artifacts and do not modify canonical trace JSONL.
