---
status: complete
completed_at: 2026-06-13T00:00:00+08:00
task: rdna4-validation-100-percent
---

# RDNA4 Validation 100 Percent Push Summary

Implemented the feasible next step toward 100% RDNA4 profiler-backed validation:
the profiler batch can now explicitly target `ready_missing_profiler_timing`
coverage rows.

What changed:

- `scripts/run_rdna4_profiler_timing_batch.py` now accepts repeatable
  `--target-status` values and passes them into target selection.
- The default selector behavior is unchanged: it still targets
  `timing_fallback`, `partial_profiler_backed`, and `profiler_blocked` unless
  an operator opts into another status.
- Tests cover both boundaries: ready-missing rows are not selected by default,
  and are selected when `target_statuses=("ready_missing_profiler_timing",)` is
  provided.

Verification:

- `uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py -q`
  passed with 33 tests.
- Latest v1.35 coverage recompute still reports 88 full profiler-backed,
  28 partial profiler-backed, 73 ready-missing profiler timing, 5 reference
  OOM-blocked, and 41 readiness-blocked problems out of the 235-problem
  denominator.
- Pure target-selection verification selected all 73 ready-missing problems,
  covering 1244 workloads.

Claim boundary:

This change does not claim strict 100% full profiler-backed timing coverage.
It removes a script-level blocker to profiling the 73 ready-missing problems.
The remaining 41 readiness-blocked and 5 reference OOM-blocked denominator rows
still require readiness fixes, policy changes, or different/larger RDNA4
hardware before full profiler-backed timing coverage can be truthfully claimed.
