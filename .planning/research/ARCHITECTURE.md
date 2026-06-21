# Research: Architecture for v1.38 Confirmed Benchmark Evidence

**Date:** 2026-06-21

## Integration Points

- `core.bench.rocm_profiler`: artifact discovery, `rocprofv3` output-format
  handling, profile result status classification, CSV/JSON/rocpd citation.
- `core.bench.profile_summary`: structured profiling metrics and diagnostic
  bottleneck hints.
- `core.scoring.amd_score` and dataset runner: official score evidence and
  aggregation policy.
- `core.scoring.baseline_artifact` or a new sibling module: measured baseline
  provenance and coverage evidence.
- `core.data.contract`: capability tokens and authority boundary statements for
  confirmed evidence, distinct from diagnostic sidecars.
- HIP-facing fixtures/docs: valid confirmed, missing score, missing baseline,
  placeholder baseline, and diagnostic-only sidecar cases.

## Data Flow

1. Evaluation emits canonical Trace JSONL and optional `rocprofv3` profile
   artifacts.
2. Profile artifact registration records files produced under the requested
   output prefix and version-specific output layouts.
3. Profile summary cites trace/profile/profiler artifacts and emits bounded
   workload/kernel metrics plus diagnostic bottleneck hints.
4. Official score evidence consumes canonical traces, SOL/SOLAR bound evidence,
   measured baseline evidence, and aggregation policy to produce a HIP-readable
   score package.
5. Measured baseline evidence records provenance and workload coverage for the
   baseline used by the score.
6. HIP cutover gate consumes SOL's confirmed evidence package and rejects runs
   with missing score, missing baseline, placeholder baseline, or insufficient
   coverage.

## Build Order

1. Fix profiler artifact registration and citations.
2. Enrich profile summary with structured metrics and bottleneck hints.
3. Add official score evidence with aggregation policy and non-null valid-run
   score.
4. Add measured baseline provenance and coverage validation.
5. Add contract/docs/fixtures tying the confirmed pass/fail gate together.
