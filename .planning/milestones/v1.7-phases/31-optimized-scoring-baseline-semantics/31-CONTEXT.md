# Phase 31: Optimized Scoring Baseline Semantics - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

This phase adds release-defined optimized scoring baseline semantics to the
existing AMD-native score workflow. It must keep canonical trace JSONL stable
while allowing derived score reports to use explicit optimized baseline timing
artifacts keyed by definition and workload UUID.

</domain>

<decisions>
## Implementation Decisions

### Baseline Artifact Contract
- Use a derived JSON artifact with schema version, release name, source, and
  entries keyed by `definition` plus `workload_uuid`.
- Store optimized baseline latency separately from PyTorch reference latency.
- Reject malformed or non-positive baseline latency values at artifact load time.
- Keep baseline artifacts out of canonical trace JSONL.

### Score Semantics
- Prefer explicit scoring baseline artifact entries when available.
- Fall back to `trace.evaluation.performance.reference_latency_ms` only as a
  provisional development fallback.
- Add `baseline_source` and a warning so reports cannot silently present
  reference latency as release-defined baseline evidence.
- Preserve existing SOL score formula and AMD claim guardrails.

### Integration Points
- Wire the artifact through `scripts/run_dataset.py --scoring-baseline` for
  opt-in AMD score reports.
- Preserve primary `sol-execbench` CLI behavior.
- Update docs and tests to show reference, candidate, scoring baseline, and SOL
  bound roles.

### the agent's Discretion
- File/module naming and exact JSON helper structure are implementation details
  as long as public schema and tests remain clear.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/scoring/amd_score.py` already builds derived
  AMD-native per-workload and suite reports.
- `scripts/run_dataset.py --amd-score-report` already connects trace JSON,
  derived AMD SOL bounds, and score output.
- `docs/internal/analysis.md` already documents derived score reports and no-equivalence
  claim boundaries.

### Established Patterns
- Derived artifacts carry schema versions, `derived: true`, evidence refs, and
  warnings.
- Public-contract changes avoid mutating canonical `Trace` models unless
  explicitly required.

### Integration Points
- Add a scoring baseline artifact loader under `core/scoring`.
- Pass optional baseline artifacts into `score_amd_native_trace_workload()`.
- Extend dataset runner tests and AMD-native score tests.

</code_context>

<specifics>
## Specific Ideas

Use the user's clarified priority: this phase addresses missing scoring
baseline implementation, not NVIDIA/B200 parity.

</specifics>

<deferred>
## Deferred Ideas

- Agentic optimizer generation of the optimized baselines is future work.
- Full original paper dataset extraction is deferred.

</deferred>
