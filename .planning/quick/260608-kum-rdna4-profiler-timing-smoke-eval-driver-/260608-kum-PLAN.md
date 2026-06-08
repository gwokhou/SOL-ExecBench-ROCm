---
quick_id: 260608-kum
slug: rdna4-profiler-timing-smoke-eval-driver-
status: in_progress
created_at: 2026-06-08
---

# Quick Task 260608-kum: Fix RDNA4 Profiler Timing Eval Driver Finalization

## Goal

Fix the profiler-backed timing smoke path so `rocprofv3` can finalize and emit
kernel trace CSV files when SOL ExecBench runs generated `eval_driver.py`.

## Tasks

1. Add profiler-friendly eval-driver finalization.
   - Files: `src/sol_execbench/driver/templates/eval_driver.py`,
     `src/sol_execbench/core/bench/rocm_profiler.py`
   - Action: keep `os._exit(0)` as the default, but allow profiler timing runs
     to request graceful `sys.exit(0)` through an environment variable.
   - Verify: eval-driver subprocess test.

2. Prefer kernel trace CSV outputs.
   - Files: `src/sol_execbench/core/bench/rocm_profiler.py`,
     `tests/sol_execbench/test_rocm_profiler.py`
   - Action: make CSV discovery prefer `_kernel_trace.csv` over agent/runtime
     CSV files.
   - Verify: profiler tests.

3. Re-run real RDNA4 smoke.
   - Action: run `scripts/run_rdna4_profiler_timing_smoke.py` outside sandbox.
   - Verify: `profiler_collected=true` and non-null CSV path.
