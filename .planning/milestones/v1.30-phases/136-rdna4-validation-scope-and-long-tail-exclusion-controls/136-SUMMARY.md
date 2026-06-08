# Phase 136 Summary

## Status

Completed on 2026-06-07.

## Delivered

- Defined the RDNA4 `gfx1200` validation scope, denominator boundary, artifact
  expectations, and claim limits in
  `136-RDNA4-VALIDATION-SCOPE.md`.
- Added a default-off `--long-tail-exclusions` configuration path to
  `scripts/run_dataset.py`.
- Added a validated long-tail exclusion schema supporting problem, workload,
  and shard exclusions with required reason and evidence refs.
- Wired long-tail exclusions into execution closure provenance, source refs,
  status totals, and stale-provenance comparison.
- Preserved excluded entries as `excluded_long_tail`; they do not count as
  passes, failures, or benchmark-grade validation authority.
- Added CPU-safe and runner-level tests for ready-subset and plain dataset
  exclusion behavior.

## Boundaries

- Phase 136 does not run full RDNA4 validation.
- Pipeline execution mode does not accept `--long-tail-exclusions` in this
  phase because closure accounting for pipeline runs is not implemented.
- Derived/timing-only reuse of existing traces is not rewritten by exclusion
  config in this phase.
- No CDNA4 real-validation artifact was found in the current repository
  context; CDNA3/MI300-family records were used only as examples for long-tail
  validation handling.
