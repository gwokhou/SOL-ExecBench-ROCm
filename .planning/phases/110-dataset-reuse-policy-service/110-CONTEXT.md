# Phase 110: Dataset Reuse Policy Service - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase moves dataset existing-trace reuse policy into a tested core helper
so `scripts/run_dataset.py` no longer owns stale-provenance/rerun/failed-trace
decision logic inline.

</domain>

<decisions>
## Implementation Decisions

### Reuse Policy Scope
- Centralize the decision for existing trace reuse vs rerun in
  `sol_execbench.core.dataset`.
- Keep trace parsing, summary generation, derived report extension, and closure
  record construction in the script for this phase.
- Preserve existing user-facing messages and behavior.

### Inputs
- The policy helper should take explicit inputs: rerun flag, traces path,
  failed trace count, optional execution closure path, and current provenance.
- If no execution closure path is configured, preserve default resume behavior:
  reuse an existing all-passing trace.
- If closure provenance is configured, reuse only when prior closure provenance
  is readable and matches current provenance.

### Testing Boundary
- Add direct core helper tests for missing traces, rerun, previous failures,
  no closure path, missing closure, matching provenance, and mismatched
  provenance.
- Keep existing run-dataset integration tests passing.

### the agent's Discretion
The helper may return a dataclass or serializable structure as long as the
script can apply it without changing behavior.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `prior_closure_provenance()` and `stale_provenance_mismatch()` already live in
  `src/sol_execbench/core/dataset/run_closure.py`.
- `compare_execution_closure_provenance()` already lives in
  `src/sol_execbench/core/dataset/execution_closure.py`.
- `scripts/run_dataset.py` already delegates ready-subset selection and closure
  helpers through thin wrappers.

### Established Patterns
- Dataset helpers return simple dicts/dataclasses that scripts can serialize or
  inspect.
- Existing run-dataset tests use monkeypatches to assert whether CLI execution
  is skipped or invoked.

### Integration Points
- The existing trace reuse block starts around the `if not args.rerun and
  traces_path.exists()` branch in `scripts/run_dataset.py`.
- Provenance mismatches are accumulated into `provenance_mismatches`.

</code_context>

<specifics>
## Specific Ideas

Introduce `DatasetReuseDecision` with fields such as `should_reuse`, `reason`,
and `provenance_mismatches`.

</specifics>

<deferred>
## Deferred Ideas

- Full closure/evidence completeness refactor is Phase 111.
- Full failure-mode matrix expansion is Phase 112.
- Sharding is Phase 113.

</deferred>
