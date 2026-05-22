# Phase 29: Derived AMD Scoring Workflow - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 29 connects canonical trace JSONL, live timing evidence, AMD SOL bound
artifacts, and baseline latency inputs into derived AMD-native workload and
suite score reports. It must expose the workflow additively and must not change
canonical traces, public schemas, primary CLI defaults, or the SOL score
formula.

</domain>

<decisions>
## Implementation Decisions

### Scoring Workflow Surface
- Add a reusable score workflow builder and a dataset runner additive option.
- Use `--amd-score-report <path>` as the dataset runner output option; only
  write derived AMD score reports when the option is provided.
- Let core scoring builders accept lightweight evidence reference mappings.
- Emit one suite JSON report with workload scores, warnings, and evidence refs.

### Missing Evidence Semantics
- Missing timing, baseline, or bound evidence produces an unscored guarded state
  instead of inventing a score.
- Unsupported operations can still produce a provisional score when numeric
  inputs exist, but the report must carry warnings.
- Unvalidated hardware produces a provisional score plus warning rather than
  blocking scoring.
- Evidence references should include trace, timing, SOL-bound, baseline, and
  hardware-model refs when available.

### Contract and Test Boundary
- Do not modify trace JSONL or trace models.
- Do not modify `sol_score()`.
- Test core workflow builder, dataset runner additive output, missing evidence
  guards, and trace immutability.
- Update analysis or dataset usage docs to mark AMD-native score reports as
  derived artifacts.

### the agent's Discretion
All implementation details not fixed above are at the agent's discretion, with
public contract preservation as the hard constraint.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/scoring/amd_score.py` already builds workload and
  suite score report dataclasses.
- `src/sol_execbench/core/scoring/amd_sol.py` builds SOL bound artifacts and
  coverage summaries.
- `scripts/run_dataset.py` already writes per-problem traces and a summary.

### Established Patterns
- Derived reports carry `schema_version`, `derived`, and `canonical_output`.
- Guardrails are warnings, not trace mutations.
- Dataset runner options are additive and disabled unless requested.

### Integration Points
- Extend `amd_score.py` with workflow helper functions.
- Extend `scripts/run_dataset.py` with `--amd-score-report`.
- Extend `tests/sol_execbench/test_amd_native_score.py` and add focused script
  tests as needed.

</code_context>

<specifics>
## Specific Ideas

For dataset runner reference-solution runs, `evaluation.performance.latency_ms`
can be measured timing and `reference_latency_ms` can be used as baseline input
when present. SOL bound artifacts can be derived from problem definitions and
workloads at report-generation time.

</specifics>

<deferred>
## Deferred Ideas

- Making AMD score reports a primary `sol-execbench` default output.
- Changing the SOL score formula.
- Requiring fixed artifact directory layouts for all evidence.

</deferred>
