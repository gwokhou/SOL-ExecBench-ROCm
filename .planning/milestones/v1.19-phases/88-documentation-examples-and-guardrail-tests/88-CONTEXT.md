# Phase 88: Documentation, Examples, And Guardrail Tests - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase completes v1.19 by documenting and testing how researchers should
generate, inspect, and safely interpret the new evidence surfaces from Phases
83-87: execution closure hardening, paper denominator accounting, Matrix schema
export, Matrix semantic diff, and AMD bound sanity.

The work is docs, examples/fixtures, and CPU-safe guardrails only. It must not
expand hardware validation, run GPU/Docker probes, relock dependencies, or
change canonical benchmark contracts.

</domain>

<decisions>
## Implementation Decisions

### Documentation Entry
- Add a centralized v1.19 evidence guide as the primary researcher-facing
  entry point.
- The guide should explain how to generate and interpret:
  execution closure, paper denominator reports, Matrix schema exports, Matrix
  semantic diffs, and AMD bound sanity reports.
- Update `docs/user/CLAIMS.md`, `docs/user/TESTING.md`, and/or `docs/user/RESEARCHER-GUIDE.md`
  only enough to point to the guide and preserve claim boundaries.

### Fixture Examples
- Add small fixture/example reports with obvious demo paths, relative refs,
  checksums, bounded log refs, and authority-false or diagnostic-only claim
  boundaries.
- Do not use real RDNA 4 performance numbers or real hardware validation output
  as public examples.
- Fixture examples should demonstrate artifact shape and interpretation, not
  benchmark success.

### Guardrail Test Strength
- Add CPU-safe docs/example wording tests covering the centralized guide,
  `CLAIMS.md`, `TESTING.md`, and fixture examples.
- Required negative statements: no full 235-problem paper validation, no
  upstream SOLAR parity, no score authority, no leaderboard readiness, no
  MI300X-on-CDNA3 and CDNA4 validation, no native-host ROCm Matrix validation, and no
  new-hardware validation.
- Avoid broad full-doc scans that create noisy false positives; focus on public
  v1.19 entry points and example artifacts.

### the agent's Discretion
- The agent may choose the exact guide filename, but it should be discoverable
  from existing docs and tests.
- The agent may generate fixture reports by hand or through existing helpers,
  provided outputs are deterministic, small, and clearly demo-only.
- Prefer extending existing docs guardrail tests over creating brittle global
  scans.

</decisions>

<code_context>
## Existing Code And Docs Insights

### Reusable Docs
- `docs/user/CLAIMS.md` already contains project-wide claims and evidence
  boundaries, including Docker Matrix/native-host distinctions and claim
  upgrade rules.
- `docs/user/TESTING.md` already documents CPU-safe test groups, ROCm Matrix
  guardrails, and live ROCm validation boundaries.
- `docs/user/RESEARCHER-GUIDE.md` already gives researcher workflows and artifact
  interpretation tables, but it does not yet summarize all v1.19 evidence
  surfaces.
- `docs/internal/analysis.md` and older release closure docs contain boundary language
  that can be reused carefully without overclaiming.

### Reusable Tests
- `tests/sol_execbench/test_public_contract_guardrails.py` already guards
  canonical contracts, public CLI exposure, v1.19 sidecar-only reports, and
  important claim-boundary strings.
- Existing focused tests cover the actual implementations:
  - `test_execution_closure_contract.py`
  - `test_paper_denominator_report.py`
  - `test_paper_denominator_script.py`
  - `test_matrix_schema_export.py`
  - `test_matrix_semantic_diff.py`
  - `test_run_dataset_execution_closure.py`
  - `test_amd_bound_sanity.py`
  - `test_amd_bound_sanity_script.py`

### Evidence Surfaces To Reference
- Phase 83/86: execution closure sidecar and dataset runner hardening.
- Phase 84: `scripts/report_paper_denominator.py`.
- Phase 85: `scripts/export_matrix_schema.py` and
  `scripts/diff_matrix_reports.py`.
- Phase 87: `scripts/report_amd_bound_sanity.py`.

</code_context>

<specifics>
## Specific Ideas

Plan should produce one concise evidence guide, a small examples directory or
fixture set for v1.19 report shapes, focused docs/example guardrail tests, and
testing docs that list the CPU-safe command covering v1.19 surfaces. The tests
should verify wording and fixture shape, not execute GPU/Docker validation.

</specifics>

<deferred>
## Deferred Ideas

- Full 235-problem paper validation, upstream SOLAR equivalence comparison,
  AMD SOL/SOLAR model validation, leaderboard readiness, CDNA3-family including MI300X, CDNA 4,
  native-host validation, new hardware validation, Docker privilege changes,
  and dependency relocking remain out of scope.
- Real RDNA 4 performance example publication remains out of scope for Phase
  88; fixtures should remain demo-only.

</deferred>
