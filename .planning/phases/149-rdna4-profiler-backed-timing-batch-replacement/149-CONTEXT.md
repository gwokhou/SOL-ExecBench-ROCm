# Phase 149 Context: RDNA4 Profiler-Backed Timing Batch Replacement

## Goal

Replace the 121 RDNA4 fallback timing sidecars with explicit `rocprofv3`
profiler-backed kernel activity timing sidecars where the Phase 148 coverage
ledger reports `timing_fallback`.

## Starting Point

- Phase 148 generated the current coverage baseline:
  - 235 problem denominator
  - 0 profiler-backed problems
  - 121 fallback timing problems
  - 114 readiness-blocked problems
- The previous smoke fix proved that direct staged `eval_driver.py` profiling
  works when `SOL_EXECBENCH_GRACEFUL_EXIT=1`.
- The old dataset timing path profiles source-policy timing and leaves PyTorch
  reference solutions on fallback/device-events; Phase 149 needs an explicit
  forced `rocprofv3` batch path.

## Constraints

- Full 121-problem profiling is a long RDNA4 hardware job. The runner must be
  resumable and support small limits for verification.
- Replacement sidecars must not be written for failed profiler collections.
- Default benchmark timing policy must remain unchanged; forced profiler
  replacement should be opt-in through the Phase 149 script.
