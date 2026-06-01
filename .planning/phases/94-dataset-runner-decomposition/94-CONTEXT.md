# Phase 94: Dataset Runner Decomposition - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure/refactor phase)

<domain>
## Phase Boundary

This phase decomposes `scripts/run_dataset.py` so dataset selection,
run-state/resume decisions, closure/provenance construction, derived evidence
discovery, and output persistence move into tested package helpers under
`src/sol_execbench/core/dataset/`. The script should remain a CLI orchestration
wrapper and preserve existing command-line behavior, filenames, trace semantics,
and public sidecar contracts.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- All implementation choices are at the agent's discretion because this is a
  pure infrastructure/refactor phase.
- Preserve public CLI behavior and sidecar schemas unless a plan explicitly
  proves an existing behavior is invalid.
- Prefer small typed helpers in `src/sol_execbench/core/dataset/` over new
  abstractions in `scripts/`.
- Keep extraction incremental: move cohesive logic with focused tests rather
  than rewriting the entire runner state machine at once.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/dataset/execution_closure.py` already owns closure
  status vocabulary, provenance comparison, report construction, and report
  writing.
- `src/sol_execbench/core/dataset/evidence_refs.py` already owns bounded
  derived evidence refs and safe sidecar stem behavior.
- Existing dataset modules cover layout, inventory, readiness, ready subsets,
  paper denominator, parity gaps, manifests, checksums, and categories.

### Established Patterns
- Dataset-side reusable behavior belongs under
  `src/sol_execbench/core/dataset/` with focused tests under
  `tests/sol_execbench/`.
- Public sidecars use bounded relative refs, stable status/reason vocabulary,
  and deterministic JSON/Markdown where applicable.
- Optional evidence and report helpers should degrade explicitly rather than
  silently changing benchmark authority.

### Integration Points
- `scripts/run_dataset.py` is the main orchestration entry point for single
  problem and dataset category runs.
- `tests/sol_execbench/test_run_dataset_execution_closure.py` and related
  run_dataset tests are the primary regression surface for resume/closure
  behavior.
- `.planning/codebase/CONCERNS.md` tracks the remaining script-accretion risk
  and should be updated in the final boundary phase, not during this phase
  unless a targeted concern is fully closed.

</code_context>

<specifics>
## Specific Ideas

No product-facing behavior changes. Use the roadmap success criteria and
existing tests as the contract.

</specifics>

<deferred>
## Deferred Ideas

- Full dataset-scale execution and hardware validation remain outside this
  phase.
- Moving derived evidence generation into a fully separate post-processing CLI
  is future work unless it falls out naturally from helper extraction.

</deferred>
