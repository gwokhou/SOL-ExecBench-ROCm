# Phase 148 Summary: RDNA4 Profiler-Backed Timing Coverage Closure

**Status:** Completed 2026-06-08.

## Delivered

- Added `sol_execbench.core.dataset.profiler_timing_coverage`, a
  denominator-aware coverage model for profiler-backed RDNA4 timing evidence.
- Added `scripts/run_rdna4_profiler_timing_coverage.py` to generate JSON and
  Markdown reports over `data/SOL-ExecBench/benchmark` and timing sidecar
  directories.
- Added CPU-safe tests that preserve the key claim boundary: fallback timing
  sidecars remain visible but do not count as profiler-backed `rocprofv3`
  coverage.
- Generated the first current-state coverage baseline under
  `out/rdna4-profiler-timing-coverage/`.

## Baseline Result

- 235 problem denominator accounted.
- 0 profiler-backed timing problems.
- 121 fallback timing problems.
- 114 readiness-blocked problems.
- 0 ready problems missing any timing sidecar.

## Residual Work

Profiler-backed coverage still requires expanded RDNA4 `rocprofv3` batch
execution using the direct staged `eval_driver.py` path. Phase 148 only created
the denominator ledger and baseline; it does not upgrade timing authority.
