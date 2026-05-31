# Phase 83: Closure Contracts And Provenance Foundation - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase creates a strict, deterministic execution-closure sidecar contract
and provenance helper layer for dataset execution evidence. It extracts the
closure vocabulary and serialization policy currently concentrated in
`scripts/run_dataset.py` into reusable core helpers, while preserving canonical
Trace JSONL, correctness, timing, score, and evaluator behavior.

</domain>

<decisions>
## Implementation Decisions

### Contract Shape
- Use a sidecar-only `sol_execbench.execution_closure.v1` contract; do not add
  closure fields to canonical Trace, Definition, Workload, Solution, score, or
  evaluator schemas.
- Prefer Pydantic v2 models or narrow typed dataclasses where the rest of
  `src/sol_execbench/core/dataset/` already uses strict model contracts and
  deterministic `to_json()` helpers.
- Preserve the existing public status vocabulary as the compatibility baseline:
  `attempted_passed`, `attempted_failed`, `not_attempted`, `filtered`,
  `skipped_existing_pass`, `missing_trace`, and `derived_evidence_missing`.
- Add only the minimum new reason/provenance structures needed for later
  denominator and runner-hardening phases; avoid reworking dataset execution
  flow in this phase.

### Provenance Policy
- Provenance checks should compare manifest, readiness, ready-subset, workload
  identity, solution mode, and requested evidence requirements before existing
  traces are treated as reusable.
- The phase may expose comparison helpers and mismatch diagnostics, but
  `scripts/run_dataset.py` integration should stay minimal until Phase 86.
- Use relative refs/checksums where practical and keep sensitive or noisy data
  out of sidecars.
- Deterministic ordering and totals are required so generated reports are
  reviewable in tests and commits.

### Testing Strategy
- Add CPU-safe unit tests around closure models, status validation,
  deterministic serialization, totals, and provenance mismatch detection.
- Keep existing `tests/sol_execbench/test_run_dataset_execution_closure.py`
  behavior green; treat it as compatibility coverage for current runner
  outputs.
- Add source/contract guardrails if needed to prove closure fields remain
  sidecar-only and do not mutate canonical trace semantics.
- Do not introduce live ROCm, Docker, CDNA 3, MI300X, CDNA 4, or native-host
  validation requirements.

### the agent's Discretion
- The agent may choose exact module names, helper boundaries, and model
  factoring, with a strong preference for a focused
  `src/sol_execbench/core/dataset/execution_closure.py` module as suggested by
  project research.
- The agent may decide whether to export the new helpers from
  `src/sol_execbench/core/dataset/__init__.py` if doing so matches existing
  package patterns.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/dataset/checksums.py` provides stable checksum
  helpers used by dataset sidecars.
- `src/sol_execbench/core/dataset/readiness.py` and
  `src/sol_execbench/core/dataset/ready_subset.py` show existing Pydantic
  sidecar models, checksum fields, and deterministic write helpers.
- `src/sol_execbench/core/dataset/parity_gap.py` already consumes execution
  closure dictionaries and can guide compatibility expectations.
- `tests/sol_execbench/test_run_dataset_execution_closure.py` contains current
  fixture coverage for runner-produced execution closure payloads.

### Established Patterns
- Dataset sidecars use explicit `schema_version` constants, strict status and
  reason vocabularies, checksum fields, and deterministic JSON serialization.
- Core modules should be mostly side-effect free and return typed models or
  JSON-serializable payloads rather than printing.
- Large orchestration code in `scripts/run_dataset.py` should be changed
  carefully and only through narrow helper calls.

### Integration Points
- Current closure helpers live around lines 623-954 and runner integration
  around the ready-subset and execution-closure paths in
  `scripts/run_dataset.py`.
- Future phases will consume this phase from denominator accounting,
  dataset-runner hardening, and AMD bound sanity reports.
- Public contract guardrails live in
  `tests/sol_execbench/test_public_contract_guardrails.py`.

</code_context>

<specifics>
## Specific Ideas

Implement the core closure contract first, then optionally adapt
`scripts/run_dataset.py` to delegate status/totals/record construction to the
new helpers only where it is low-risk. Leave deeper resume/reuse enforcement
for Phase 86.

</specifics>

<deferred>
## Deferred Ideas

- Full `scripts/run_dataset.py` resume/reuse enforcement is deferred to Phase
  86.
- Paper denominator rollups are deferred to Phase 84.
- AMD SOL/SOLAR sanity evidence is deferred to Phase 87.
- Any new hardware validation remains out of scope for v1.19.

</deferred>
