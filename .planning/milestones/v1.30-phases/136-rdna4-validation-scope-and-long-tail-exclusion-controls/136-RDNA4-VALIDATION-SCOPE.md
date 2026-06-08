# Phase 136 RDNA4 Validation Scope

## Purpose

Phase 136 defines the RDNA4 validation denominator and runner controls needed before
benchmark-grade execution begins. It does not complete RDNA4 validation by itself.
Full RDNA4 execution is reserved for Phase 138, and final claim closure is reserved
for Phase 141.

## Validation Target

- Architecture target: RDNA4 `gfx1200`.
- Software baseline: ROCm >= 7.0.
- Dataset source: local SOL ExecBench benchmark assets, expected under
  `data/SOL-ExecBench/benchmark` or an equivalent user-provided path.
- Dataset assets are not redistributed by this repository.
- Execution artifacts should stay under an explicit output root such as
  `out/rdna4-*`, with sidecars kept together for provenance review.

## Denominator Boundary

The benchmark-grade denominator is the discovered local SOL ExecBench problem and
workload set after documented dataset readiness filters, not a post-hoc set of only
successful workloads. A validation claim must report:

- discovered problems and workloads;
- selected categories and workload caps;
- readiness exclusions and blockers;
- explicit long-tail exclusions;
- attempted, passed, failed, timed-out, skipped, and evidence-gap records;
- source refs and checksums for all selection sidecars.

## Long-Tail Exclusion Contract

`scripts/run_dataset.py --long-tail-exclusions <path>` is a default-off escape hatch
for known super-long-tail validation shards. It is intentionally narrow:

- exclusions must be explicit JSON entries scoped to `problem`, `workload`, or
  `shard`;
- every entry must carry a reason and evidence reference;
- excluded workloads are recorded as `excluded_long_tail`;
- exclusions are not passes, not failures, and not evidence of paper parity;
- exclusions do not authorize leaderboard or benchmark-grade claims unless the
  final claim boundary says so explicitly;
- derived/timing-only reuse of existing traces is outside this phase's exclusion
  rewrite scope.

## Prior Validation Records

The current repository contains CDNA3/MI300-family validation and readiness records
that are relevant as examples for long-tail handling:

- `.planning/milestones/CDNA3-VALIDATION-HANDOFF.md`
- `docs/internal/cdna3_gfx942_validation_attempt.md`
- `docs/internal/cdna3_validation_readiness.md`
- `docs/internal/mi300x_validation_readiness.md`
- `.planning/quick/260604-cdna3-gfx942-validation-attempt-record/KNOWN_ISSUES.md`

No exact CDNA4 real-validation record was found in the current repository context.
CDNA4 remains a deferred or unavailable reference point unless a concrete artifact
is added later.

## Handoff To Later Phases

Phase 137 should turn this scope into an executable RDNA4 runbook with health checks,
polling cadence, and resume rules for jobs that may run for many hours. Phase 138
should run the RDNA4 validation itself and produce the execution closure sidecars
needed for Phase 141 claim review.
