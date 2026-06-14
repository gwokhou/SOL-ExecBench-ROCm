---
quick_id: 260613-x39
slug: regenerate-rdna4-validation-completeness
description: Regenerate RDNA4 validation completeness report from scripts/internal
status: in_progress
---

# Quick Task: Regenerate RDNA4 Validation Completeness Report

Rebuild the RDNA4 validation completeness artifacts from the internal RDNA4
report scripts, using the latest plus-L2/041 evidence overlay as the canonical
output root.

## Scope

- Re-run `scripts/internal/rdna4/run_rdna4_profiler_timing_coverage.py`.
- Re-run `scripts/internal/rdna4/run_rdna4_profiler_sharded_closure.py`.
- Refresh the canonical merged coverage, blocker ledger, sharded closure audit,
  and evaluation summary under
  `out/rdna4-validation-reeval-20260613-latest-plus-l2041/merged/`.

## Verification

- Inspect regenerated summary counts.
- Confirm generated JSON files parse successfully.
