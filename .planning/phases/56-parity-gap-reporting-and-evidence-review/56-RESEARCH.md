# Phase 56: Parity Gap Reporting And Evidence Review - Research

**Researched:** 2026-05-23
**Domain:** deterministic sidecar aggregation, JSON/Markdown reporting, parity claim boundaries
**Confidence:** HIGH

## Summary

Phase 56 should add a small sidecar aggregation layer rather than extending the
benchmark runner. The inputs already exist from Phases 53-55: manifest,
inventory, readiness, ready subset, execution closure, and optional derived
score/evidence sidecars. The report can be generated deterministically with
stdlib JSON and Markdown string formatting; no new package is needed.

Primary implementation target:

- `src/sol_execbench/core/dataset/parity_gap.py` for pure aggregation helpers,
  report models, JSON serialization, and Markdown rendering.
- `scripts/report_parity_gaps.py` for a thin CLI over the helper.
- `tests/sol_execbench/test_parity_gap_report.py` for fixture coverage.
- `docs/analysis.md` and public guardrails for claim wording.

## Requirements Map

| Requirement | Implementation Direction |
|-------------|--------------------------|
| GAP-01 | Load acquisition, inventory, readiness, and execution closure JSON sidecars and emit JSON + Markdown reports. |
| GAP-02 | Aggregate category and suite denominators: discovered, parsed, ready, blocked, not attempted, skipped, attempted, passed, failed, scored, degraded, unscored. |
| GAP-03 | Group readiness and execution blockers by stable reason code with next actions from readiness records. |
| GAP-04 | Count evidence completeness from execution closure refs and AMD-native report refs: trace, timing, AMD score, AMD SOL, SOLAR derivation. |
| GAP-05 | Store deterministic source artifact refs and reject absolute/path-traversing output refs where the report controls paths. |

## Existing Inputs

- `DatasetManifest` includes selected categories, layout diagnostics,
  category counts/checksums, claim boundary, and manifest checksum.
- `DatasetInventory` includes category/problem/workload records, suite
  denominators, diagnostics, and inventory checksum.
- `DatasetReadiness` includes per-workload statuses, stable reason codes,
  next actions, and readiness checksum.
- `ReadySubset` includes ready problem/workload refs and ready-subset checksum.
- `execution_closure.json` from Phase 55 includes per-workload closure status,
  readiness metadata, trace refs, derived evidence refs/gaps, provenance, and
  claim boundary.
- AMD-native score reports include `scored_count`, `unscored_count`, per-score
  `supported`, warnings, `evidence_refs`, and `derived_evidence_refs`.

## Recommended Report Shape

Top-level JSON:

```json
{
  "schema_version": "sol_execbench.parity_gap_report.v1",
  "created_at": "2026-05-23T00:00:00Z",
  "sources": {},
  "suite": {},
  "categories": [],
  "blockers": [],
  "evidence_completeness": {},
  "claim_boundary": {}
}
```

Category/suite denominator keys should be stable and include all required
counts even when zero:

- `discovered`
- `parsed`
- `ready`
- `blocked`
- `not_attempted`
- `skipped`
- `attempted`
- `passed`
- `failed`
- `scored`
- `degraded`
- `unscored`

Blocker rows should include `reason_code`, `count`, `categories`,
`example_refs`, and `next_actions`.

Evidence completeness should count:

- `trace`
- `timing`
- `amd_native_score`
- `amd_sol`
- `solar_derivation`

## Validation Strategy

- Fixture test with one parsed/ready/passed/scored workload and one blocked
  workload.
- Determinism test: same fixed timestamp yields byte-identical JSON and
  Markdown.
- CLI test: script writes both JSON and Markdown paths.
- Guardrail test: report and docs wording state bounded reporting is not full
  validation, paper parity, upstream SOLAR parity, or leaderboard readiness.

## Risks

- Score report category information may be absent. Use execution closure and
  inventory/readiness to map workload UUIDs to category/problem when possible;
  count unmatched scores at suite level with an `unknown` category only if
  necessary.
- Derived sidecar directories can be large. Do not recursively inspect all
  sidecars in Phase 56 unless needed; prefer refs already present in execution
  closure and AMD score report.
- Missing optional artifacts should be evidence gaps, not fatal errors.

## Research Complete
