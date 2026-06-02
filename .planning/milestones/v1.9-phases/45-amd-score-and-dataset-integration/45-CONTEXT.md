# Phase 45: AMD Score And Dataset Integration - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 45 wires Phase 44 AMD SOL bound artifact v2 sidecars into AMD-native
workload and suite score reports, plus optional dataset-run sidecar emission.

This phase keeps canonical trace JSONL unchanged. It may extend derived
`scripts/run_dataset.py --amd-score-report` behavior because that script-level
report is already a derived workflow outside the primary `sol-execbench` CLI.
</domain>

<decisions>
## Implementation Decisions

### Score Integration
- AMD-native score reports should consume `AmdSolBoundV2Artifact` as the
  preferred bound evidence.
- Keep v1 `AmdSolBoundArtifact` compatibility where existing callers use it,
  but new tests and dataset report code should exercise v2.
- Preserve evidence references for trace, timing, SOL-bound sidecar, baseline,
  and hardware model.
- V2 `aggregate_bound.status == "unscored"` must produce a workload score with
  `score=None` and deterministic warnings.
- V2 degraded evidence may still compute a provisional derived score, but
  warning strings must expose the degraded state.

### Dataset Integration
- `scripts/run_dataset.py --amd-score-report` should build v2 SOL bound
  sidecars for each workload before scoring.
- Add optional sidecar emission for dataset runs through a script-level option,
  scoped to `--amd-score-report`, without changing primary trace JSONL output.
- When sidecars are emitted, score evidence refs should point to the emitted
  v2 artifact path; otherwise use a deterministic derived reference string.

### Suite Reporting
- Suite reports already expose scored/unscored counts; retain that behavior and
  add lightweight evidence summaries if needed for SCORE-04.
- Baseline summary should remain present when the dataset report is built from
  `--scoring-baseline`.

### the agent's Discretion
Exact helper names, payload summary shape, and path layout for emitted sidecars
are at the agent's discretion if they are deterministic, tested, and do not
modify canonical traces.
</decisions>

<code_context>
## Existing Code Insights

- `src/sol_execbench/core/scoring/amd_score.py` currently scores v1
  `AmdSolBoundArtifact` and already tracks evidence refs, baseline source,
  scored/unscored counts, and warnings.
- `src/sol_execbench/core/scoring/amd_sol_v2.py` now provides v2 sidecars with
  aggregate status, warnings, hardware model references, and coverage.
- `scripts/run_dataset.py` currently builds v1 artifacts inside
  `build_amd_score_reports_for_problem()` when `--amd-score-report` is set.
- `tests/sol_execbench/test_amd_native_score.py` and
  `tests/sol_execbench/test_run_dataset_amd_score.py` cover derived score
  reports and dataset helper behavior.
</code_context>

<specifics>
## Specific Ideas

- Add v2-specific scoring helpers while keeping existing v1 function names
  compatible through a union input where practical.
- Propagate v2 artifact warning strings into workload score warnings instead
  of mapping everything to a single generic warning.
- Add dataset helper argument such as `sol_bound_artifact_dir` and write
  `<definition>/<workload_uuid>.amd-sol-v2.json` or an equivalent deterministic
  filename under the requested directory.
</specifics>

<deferred>
## Deferred Ideas

- User-facing docs for score semantics belong to Phase 46.
- RDNA 4 sample-run evidence belongs to Phase 46.
- Full MI300X-on-CDNA3 validation remains deferred.
</deferred>
