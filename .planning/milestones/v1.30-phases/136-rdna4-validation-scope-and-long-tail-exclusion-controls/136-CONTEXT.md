# Phase 136: RDNA4 Validation Scope and Long-Tail Exclusion Controls - Context

**Gathered:** 2026-06-07
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase defines the RDNA4 `gfx1200` benchmark-grade validation denominator
and adds default-off, explicit `scripts/run_dataset.py` configuration for
temporarily excluding known super-long-tail shards, problems, or workloads from
execution while preserving denominator, readiness, execution-closure, and
evidence-bundle visibility.

It does not run the full RDNA4 dataset validation itself; that is Phase 138.
It does not upgrade public RDNA4 claims; that is Phase 141.
</domain>

<decisions>
## Implementation Decisions

### Infrastructure Defaults
- Treat this as an infrastructure/configuration phase; implementation choices
  are at the agent's discretion within the roadmap and requirements.
- Exclusion controls must be explicit, auditable, reversible, and default-off.
- Excluded entries must not count as passed validation and must not disappear
  from denominator or closure reports.
- The phase should verify the exact prior validation record names before naming
  any CDNA3/CDNA4 records in committed config or docs.

### the agent's Discretion
The agent may choose the specific config schema and CLI flag shape, provided
the result is deterministic, testable, and consistent with existing dataset
manifest/readiness/closure conventions.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/run_dataset.py` is the operational dataset runner CLI and should
  remain a compatibility wrapper where reusable behavior can live in
  `src/sol_execbench/core/dataset/`.
- `src/sol_execbench/core/dataset/run_state.py` owns problem discovery,
  workload selection, trace mapping, and closure status decisions.
- `src/sol_execbench/core/dataset/run_closure.py` owns run closure records,
  totals, provenance, and report writing.
- `src/sol_execbench/core/dataset/readiness.py` and `ready_subset.py` already
  model dataset readiness and selected-denominator metadata.
- `src/sol_execbench/core/dataset/evidence_refs.py` centralizes stable sidecar
  and relative reference helpers.

### Established Patterns
- Machine-readable artifacts use deterministic JSON output, usually sorted keys,
  indentation, and trailing newlines.
- CLI-facing misuse should fail with concise user-visible errors; importable
  helpers should use typed models or dataclasses where appropriate.
- Hardware validation claims are bounded and separated from schema/readiness
  support.
- Dataset redistribution and provenance boundaries must remain explicit.

### Integration Points
- The exclusion config should integrate into `scripts/run_dataset.py` argument
  parsing and the importable dataset execution path before workload selection
  and closure construction.
- Denominator, readiness, execution-closure, and evidence-bundle outputs should
  reference the exclusion config and list excluded entries explicitly.
- Tests should live next to existing dataset runner/readiness/closure coverage
  under `tests/sol_execbench/`.
</code_context>

<specifics>
## Specific Ideas

- Include config refs in generated reports so reviewers can see which long-tail
  entries were excluded and why.
- Preserve default benchmark semantics when no exclusion config is supplied.
- Prefer accepted-status names such as `excluded_long_tail` or a similarly
  explicit non-pass state over reusing pass/skip states ambiguously.
</specifics>

<deferred>
## Deferred Ideas

- Full RDNA4 dataset execution is deferred to Phase 138.
- Locked-clock timing and profiler evidence are deferred to Phase 139.
- Derived reports and final evidence bundle are deferred to Phase 140.
- Public claim closure is deferred to Phase 141.
</deferred>
