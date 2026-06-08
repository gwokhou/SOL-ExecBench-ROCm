# Phase 165 Context: RDNA4 Coverage Recompute

## Goal

Regenerate RDNA4 coverage under the hardened denominator and classifier policy
so completion numbers are reproducible and claim-safe.

## Depends On

- Phase 164: RDNA4 memory readiness classifier hardening

## Scope

- Recompute coverage over the 235-problem denominator.
- Emit included, profiler-backed, fallback, blocked, and readiness-blocked
  totals.
- Produce a machine-readable exclusion/blocker ledger with source artifacts and
  checksums.

## Primary Deliverable

- Fresh RDNA4 coverage report and deterministic coverage regression tests.
