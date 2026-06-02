# Phase 84: Paper Denominator Accounting And Claim Boundaries - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase creates a deterministic `paper_denominator_report.v1` sidecar and
Markdown summary for researcher review. It accounts for public benchmark
denominator status across inventory, readiness, ready-subset, execution
closure, AMD score, AMD SOL, and SOLAR evidence refs without treating the
accounting as paper validation, leaderboard authority, upstream SOLAR parity,
native-host validation, or new-hardware validation.

</domain>

<decisions>
## Implementation Decisions

### Report Boundary And Claims
- Generate a strict sidecar report plus deterministic Markdown summary; do not
  add fields to canonical Trace, Definition, Workload, Solution, score,
  correctness, timing, or evaluator schemas.
- Keep paper parity, upstream SOLAR parity, leaderboard authority,
  native-host validation, and new-hardware validation explicitly false.
- Treat the report as denominator accounting and evidence-gap review only, not
  full 235-problem paper validation or benchmark score authority.
- Preserve v1.19 scope: no CDNA3-family including MI300X, CDNA 4, native-host, Docker
  privilege, dependency relock, or new hardware validation expansion.

### Status Classification And Evidence
- Use stable denominator reason codes for ready, blocked, unsupported,
  deferred, evidence-missing, attempted-passed, attempted-failed, filtered,
  skipped, and not-attempted states.
- Roll up counts by problem, workload, category, readiness status, closure
  status, and evidence gap.
- Reference source manifest, inventory, readiness, ready-subset, execution
  closure, AMD score, AMD SOL, and SOLAR artifacts by bounded path/ref and
  checksum instead of duplicating full payloads.
- When evidence is absent or partial, classify it as evidence-missing or
  deferred with next-evidence hints rather than upgrading claims.

### Integration And Compatibility
- Add focused denominator helper/report code that reuses existing dataset
  sidecar patterns and Phase 83 execution closure helpers where practical.
- Prefer `src/sol_execbench/core/dataset/` for core models and builders, with
  a small script or exported helper only if it follows existing reporting
  conventions.
- Preserve current runner defaults; do not introduce resume/reuse enforcement
  here because that belongs to Phase 86.
- Add CPU-safe tests for deterministic serialization, strict models, reason
  code mapping, source refs/checksums, Markdown wording, and public contract
  guardrails.

### the agent's Discretion
- The agent may choose exact module names, CLI/script shape, and whether to
  export helpers from `src/sol_execbench/core/dataset/__init__.py`, provided
  the implementation remains narrow and consistent with existing dataset
  report modules.
- The agent may choose priority rules for combining readiness and closure
  signals, but those rules must be deterministic and must not convert missing
  evidence into validation authority.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/dataset/checksums.py` provides stable JSON checksum
  helpers used by sidecar reports.
- `src/sol_execbench/core/dataset/inventory.py` carries problem/workload
  denominator inventory data and `inventory_checksum`.
- `src/sol_execbench/core/dataset/readiness.py` and
  `src/sol_execbench/core/dataset/ready_subset.py` provide readiness status,
  reason codes, claim boundaries, and deterministic sidecar serialization.
- `src/sol_execbench/core/dataset/execution_closure.py` provides strict
  closure records, status vocabulary, provenance, totals, and deterministic
  checksum handling from Phase 83.
- `src/sol_execbench/core/dataset/parity_gap.py` already demonstrates a
  report builder that references source sidecars by path/checksum and renders
  bounded Markdown claim wording.

### Established Patterns
- Dataset reports use explicit schema-version constants, Pydantic v2 models,
  strict extra-field behavior where new contracts require it, stable ordering,
  `with_checksum()` helpers, and `write_*` functions.
- Scripts under `scripts/` are thin argparse wrappers over core builders, as
  seen in `scripts/inspect_dataset.py` and `scripts/report_parity_gaps.py`.
- Guardrail tests verify that new sidecar evidence does not mutate public
  canonical contracts or overstate paper/leaderboard validation.

### Integration Points
- Add denominator tests near `tests/sol_execbench/test_parity_gap_report.py`
  and `tests/sol_execbench/test_execution_closure_contract.py`.
- Extend `tests/sol_execbench/test_public_contract_guardrails.py` only for
  sidecar-only and claim-boundary assertions.
- Future Phase 87 AMD bound sanity reports can consume denominator outputs by
  ref/checksum, so artifact references should be stable and compact.

</code_context>

<specifics>
## Specific Ideas

Mirror the existing parity-gap report ergonomics: a core `build_*_report()`, a
`render_*_markdown()` function, deterministic write helpers, and a CPU-safe
script test that writes JSON and Markdown from fixture inputs.

</specifics>

<deferred>
## Deferred Ideas

- Dataset runner resume/reuse hardening remains deferred to Phase 86.
- AMD SOL/SOLAR bound sanity aggregation remains deferred to Phase 87.
- Matrix schema export and semantic diff remain deferred to Phase 85.
- Any new hardware validation remains out of scope for v1.19.

</deferred>
