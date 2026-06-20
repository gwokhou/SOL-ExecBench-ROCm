---
quick_id: 260613-x8p
slug: automatically-complete-rdna4-validation-
description: Automatically complete RDNA4 validation completeness P0 and P1
status: in_progress
---

# Quick Task: Complete RDNA4 P0/P1 Validation Completeness

Attempt to close all latest P0 and P1 RDNA4 completeness gaps from the regenerated
report.

## Scope

- P0: rerun the 7 `profiler_blocked` targets with workload-sharded profiler
  timing.
- P1: rerun/import the 56 `partial_profiler_backed` targets with
  workload-sharded profiler timing.
- Recompute coverage, sharded closure audit, and merged evaluation summary after
  the batch attempt.

## Strategy

- Generate target manifests from
  `out/rdna4-validation-reeval-20260613-latest-plus-l2041/merged/sharded-closure-audit.json`.
- Run `scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py` with
  `--workload-sharded`, `--max-workers 1`, `--gpu-device 0`, and bounded memory.
- Treat successful problem-level sidecars as new canonical timing evidence.
- Treat explicit OOM/runtime/profiler failures as completed diagnostic evidence
  when they cannot be closed on this host.

## Verification

- Re-run `scripts/internal/rdna4/run_rdna4_profiler_timing_coverage.py`.
- Re-run `scripts/internal/rdna4/run_rdna4_profiler_sharded_closure.py`.
- Parse regenerated JSON reports.
