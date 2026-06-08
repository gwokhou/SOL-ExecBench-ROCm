---
status: complete
---

# Phase 151 Summary

Phase 151 added OOM-safe continuation controls for RDNA4 profiler-backed timing
replacement.

## Completed

- Added explicit `--skip-problem` and `--skip-problem-file` controls to the
  replacement batch runner.
- Added `--temp-dir` so staging can be moved from root `/tmp` to the larger
  `/home` filesystem.
- Added `--mark-blocked-problem` and `--mark-blocked-only` so known OOM,
  timeout, or profiler-failure targets can be written as `profiler_blocked`
  sidecars without launching new profiler work.
- Updated resume classification so full-problem `profiler_blocked` sidecars are
  skipped while workload-limited smoke artifacts remain retryable.
- Updated profiler-failure handling to write blocked sidecars instead of
  leaving failed full-problem attempts as plain fallback timing.

## Outcome

The real RDNA4 coverage ledger now accounts for six blocked profiler targets
without increasing the strict profiler-backed count:

- `profiler_backed_problems`: `60/235`
- `partial_profiler_backed_problems`: `9`
- `profiler_blocked_problems`: `6`
- `fallback_timing_problems`: `46`

Full profiler-backed timing coverage remains false.
