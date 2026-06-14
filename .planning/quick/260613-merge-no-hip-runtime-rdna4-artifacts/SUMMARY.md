---
status: complete
---

# Summary

Promoted the latest kernel-only rocprofv3 closure coverage into:

`out/rdna4-validation-reeval-20260613-latest/merged/`

Updated canonical files:

- `evaluation-summary.json`
- `evaluation-summary.md`
- `profiler-timing-coverage-summary.json`
- `profiler-timing-coverage.json`
- `profiler-timing-blocker-ledger.json`
- `sharded-closure-audit.json`

Merged headline:

- `profiler_backed`: `96 / 235`
- `profiler_backed_coverage_pct`: `40.8511%`
- `partial_profiler_backed`: `3`
- `ready_missing_profiler_timing`: `49`
- `reference_oom_blocked`: `46`
- `readiness_blocked`: `41`
- `profiler_blocked`: `0`

Added source timing input:

`out/rdna4-profiler-blocker-fix-20260613/no-hip-runtime-24/timing`

Validation:

- Checked merged `evaluation-summary.json`,
  `profiler-timing-coverage-summary.json`, and `sharded-closure-audit.json`
  agree on the headline and partial-target counts.
