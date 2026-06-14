---
status: complete
quick_id: 260613-promote-l2041-merged
slug: promote-l2041-merged
completed_at: 2026-06-13T20:32:00+08:00
---

# Summary

Promoted the L2/041-closed RDNA4 coverage into a new canonical merged artifact
directory:

- `out/rdna4-validation-reeval-20260613-latest-plus-l2041/merged/`

Generated/updated canonical files:

- `evaluation-summary.json`
- `evaluation-summary.md`
- `profiler-timing-coverage-summary.json`
- `profiler-timing-coverage.json`
- `profiler-timing-blocker-ledger.json`
- `sharded-closure-audit.json`
- copied static reports from the previous merged directory:
  `claim-upgrade.json`, `execution-closure.json`, `paper-denominator.json`,
  `trust-summary.json`

Final headline:

- `profiler_backed`: `131 / 235`
- `profiler_backed_coverage_pct`: `55.7447`
- `partial_profiler_backed`: `0`
- `ready_missing_profiler_timing`: `0`

Verified that `L2/041_kv_shared_attention_with_dual_rope` is no longer present
in `remaining_sharded_closure_targets`; the list now contains only the 6
remaining `profiler_blocked` targets.
