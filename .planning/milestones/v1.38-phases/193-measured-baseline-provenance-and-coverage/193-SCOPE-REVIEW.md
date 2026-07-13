# Phase 193 Scope Review — Measured Baseline Provenance and Coverage

**Reviewed:** 2026-07-10
**Purpose:** Re-baseline Phase 193 scope before `/gsd:plan-phase 193`. A
pre-existing infrastructure module (`baseline_export.py`) already covers most of
BASE-01, so the remaining work is narrower than the raw success criteria
suggest. This review prevents re-implementing what already exists and focuses
planning on the genuine gaps (BASE-02 coverage validation + BASE-03 official
score gate integration).

## Context

Phase 192 (Official Score Evidence Contract) shipped 2026-06-21
(commit `0150256`). Its verification (`192-VERIFICATION.md`, Residual Risk)
explicitly hands off to 193:

> Measured baseline provenance is still the existing `scoring_baseline` input
> shape. Phase 193 must expand baseline evidence with trace pointers, hardware,
> ROCm/SOL version, timing policy, target identity, and workload coverage before
> HIP can validate full confirmed baseline coverage end to end.

`192-CONTEXT.md` (`<deferred>`) and `192-01-PLAN.md` (`Out Of Scope`) both
assign full measured baseline provenance + coverage validation to 193.

## Requirement Status (from `REQUIREMENTS.md`)

| Requirement | Status | Notes |
| --- | --- | --- |
| BASE-01 | Pending | Mostly covered by pre-existing `baseline_export.py` — see below |
| BASE-02 | Pending | Not implemented — primary remaining work |
| BASE-03 | Pending | Not implemented — requires official score gate integration |

## Pre-existing Infrastructure (NOT a 193 deliverable)

`src/sol_execbench/core/evidence/baseline_export.py` (introduced by commit
`c32cb9a #0 - Normalize baseline trace parsing`, well before Phase 192)
implements `export_hip_baseline_registry()`, a CLI exporter that produces a
HIP baseline registry JSON (`baseline_registry.v1` /
`sol_execbench.measured_baseline_registry.v1`).

It already records, per baseline entry:
- `trace_ref` / `artifact_ref` (trace pointer) ✓
- `provenance.hardware`, `.rocm_version`, `.sol_version`, `.target_id`,
  `.timing_policy` ✓
- `workload_key` / `workload_uuid` and suite-level `expected_workload_keys`
  (workload coverage set) ✓
- `latency_ms`, `score`, `facts.reference_latency_ms`, `facts.libs`,
  `facts.trace_line`

Suite-level `coverage_status` is computed as `"confirmed"` when the covered
workload keys equal the expected set, otherwise `"diagnostic"`.

**This module is a standalone CLI exporter. It is not the authoritative
baseline source consumed by `official_score.py`** (which still reads the
release-scoped `ScoringBaselineArtifact` / `scoring_baseline` shape). BASE-03
exists precisely to close that gap.

## Per-Requirement Re-Baseline

### BASE-01 — Measured baseline evidence record

> SOL emits measured baseline evidence with trace pointer, hardware identity,
> ROCm version, SOL version, target identity, timing policy, workload coverage,
> and generation timestamp.

**Already covered by `baseline_export.py`:** trace pointer, hardware, ROCm
version, SOL version, target identity, timing policy, workload coverage set.

**Gap:** `generation timestamp` is not recorded. `baseline_export.py` emits no
timestamp field.

**Recommended scope:** Add a `generated_at` (ISO-8601) timestamp to the
registry output. Do **not** re-model the provenance fields — reuse
`baseline_export.py`'s existing shape. This is a small additive change, not a
new evidence model.

### BASE-02 — Coverage validation with five-state reason codes

> SOL validates measured baseline coverage against target workloads and reports
> missing, stale, mismatched, or placeholder baseline blockers with stable
> reason codes.

**Not implemented.** This is the core remaining work of 193.

Current state:
- `baseline_export.py` reports only a binary `coverage_status`
  (`"confirmed"` / `"diagnostic"`) — no per-workload state classification.
- `official_score.py` (from 192) defines `MISSING_BASELINE_BLOCKER` and
  `PLACEHOLDER_BASELINE_BLOCKER`, but **no `stale` or `mismatched` codes**.
  A `rg` for `stale_baseline|mismatched_baseline|baseline_mismatch` across
  `src/` and `tests/` returns zero hits.

**Recommended scope:** Introduce a coverage-validation report that classifies
each expected workload's baseline into one of five stable states:
`confirmed`, `missing`, `stale`, `mismatched`, `placeholder` — each with a
stable reason code. Success criterion #4 requires tests for: complete
coverage, partial coverage, hardware mismatch, timing-policy mismatch, stale
trace pointer, and placeholder baseline rejection. None of these test cases
exist today (`test_baseline_comparison.py` has only 4 tests covering
win/parity/loss classification and formatting).

### BASE-03 — Provenance separated from scoring baseline artifacts

> SOL exposes measured baseline provenance separately from scoring baseline
> artifacts so HIP can verify confirmed coverage without treating reference
> latency as official baseline evidence.

**Partially covered, mostly not.**

- 192 already blocks `reference_latency_ms` and placeholder/reference baseline
  fallback for official score claims (decisions D-07/D-08 in
  `192-CONTEXT.md`), satisfying the "do not treat reference latency as
  official baseline" half.
- **Not covered:** `baseline_export.py`'s measured baseline provenance is not
  wired into the `official_score.py` evidence gate. The official score path
  still consumes the release-scoped `ScoringBaselineArtifact` shape only.
  There is no surface where HIP can verify *confirmed measured baseline
  coverage* as a precondition for a non-null official score.

**Recommended scope:** Make measured baseline provenance (from
`baseline_export.py`'s shape) the authoritative baseline source for the
official score gate, with the BASE-02 coverage-validation report as the
gate's coverage check. Keep `ScoringBaselineArtifact` as a separate
release-scoped artifact (do not merge the two schemas — BASE-03 requires
them separated).

## Suggested Actual Work List for 193

1. Add `generated_at` timestamp to `baseline_export.py` registry output
   (closes BASE-01 gap).
2. Implement a coverage-validation report with five-state classification
   (`confirmed` / `missing` / `stale` / `mismatched` / `placeholder`) and
   stable reason codes (closes BASE-02).
3. Add the six required coverage test cases (complete, partial, hardware
   mismatch, timing-policy mismatch, stale trace pointer, placeholder
   rejection) (closes BASE-02 success criterion #4).
4. Wire measured baseline provenance into the `official_score.py` gate as the
   authoritative baseline source, gated by the coverage-validation report;
   keep `ScoringBaselineArtifact` separate (closes BASE-03).

## Risks And Notes

- **Do not duplicate `baseline_export.py`.** Any 193 plan that re-creates
  provenance fields is a scope error. Reuse the existing shape and extend it.
- **Stale / mismatched detection needs source-of-truth signals.** "Stale trace
  pointer" and "hardware/timing-policy mismatch" require comparing the
  baseline's recorded provenance against the current run's environment. Confirm
  during planning where the current-run environment identity is captured
  (likely `trace.evaluation.environment` / `libs`, already read by
  `baseline_export.py`).
- **Decision sidecar is orthogonal.** The recent decision sidecar work
  (`docs/user/decision_sidecar.md`) is diagnostic-only optimization guidance and
  does not touch confirmed pass/fail authority. It does not substitute for 193
  or 194 and does not change this scope.
- **ROADMAP status was stale.** As of 2026-07-10 the ROADMAP listed 192 as
  pending; it has been corrected to `Verified 2026-06-21`.

## Next

Run `/gsd:plan-phase 193` with this review as input. The plan should treat
items 1–4 above as the task breakdown and explicitly cite
`baseline_export.py` as the reuse anchor.
