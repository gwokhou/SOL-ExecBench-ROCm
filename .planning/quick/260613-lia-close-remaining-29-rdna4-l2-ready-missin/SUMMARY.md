# Quick Task Summary: Close Remaining RDNA4 L2 Ready-Missing

Status: complete

## Outcome

The remaining 29 `L2/*` ready-missing profiler timing targets have all been
closed with explicit evidence.

- Final coverage: `out/rdna4-ready-missing-profiler-closure-20260613/coverage-stage2/`
- Valid final batch: `out/rdna4-ready-missing-profiler-closure-20260613/l2-batch-03/`
- `ready_missing_profiler_timing_problems`: 0
- `profiler_backed_problems`: 121 / 235
- `profiler_backed_coverage_pct`: 51.4894%
- `profiler_blocked_problems`: 35
- `partial_profiler_backed_problems`: 3
- `reference_oom_blocked_problems`: 35

## Notes

The task closed the accounting gap, not the profiler-backed timing gap. The 29
targets moved from `ready_missing_profiler_timing` to `profiler_blocked`.

`l2-batch-02` is excluded from the final merge because it ran inside the sandbox
without visible ROCm GPU devices. The valid resumed run is `l2-batch-03`, run
with elevated host GPU access.

Dominant blockers in the 29 target sidecars:

- 23 problems: at least one workload hit `rocprofv3` exit code `-11`.
- 5 problems: dynamic estimated timing-input cap blocked unsafe launches.
- 1 problem: `rocprofv3` produced no kernel activity rows.

Verification:

- `uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py`
  passed with `38 passed`.
- `uv run python -m py_compile scripts/run_rdna4_profiler_timing_batch.py`
  passed.
