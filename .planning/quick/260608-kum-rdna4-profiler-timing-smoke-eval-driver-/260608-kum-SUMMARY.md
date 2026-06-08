---
quick_id: 260608-kum
slug: rdna4-profiler-timing-smoke-eval-driver-
status: complete
completed_at: 2026-06-08
---

# Quick Task 260608-kum Summary

## Completed

- Root-caused the missing profiler CSV to generated `eval_driver.py` ending with
  `os._exit(0)`, which prevents `rocprofv3` finalizers from writing trace
  files.
- Added `SOL_EXECBENCH_GRACEFUL_EXIT=1` support to the eval driver template so
  profiler-backed timing runs can request normal interpreter teardown while
  ordinary validation keeps the hard-exit default.
- Updated the profiler default runner to request graceful exit.
- Updated `scripts/run_rdna4_profiler_timing_smoke.py` to stage the Triton
  example and profile `eval_driver.py` directly instead of profiling the outer
  `sol-execbench` CLI process.
- Made `rocprofv3` CSV discovery prefer `_kernel_trace.csv` over agent/runtime
  CSV files.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_rdna4_profiler_timing_smoke.py tests/sol_execbench/driver/test_eval_driver.py::test_eval_driver_supports_profiler_graceful_exit_switch -q`
  - `24 passed in 4.36s`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check ...`
  - `All checks passed!`
- Real RDNA4 smoke:
  - Command:
    `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_rdna4_profiler_timing_smoke.py --output-dir out/rdna4-profiler-backed-timing-smoke --timing-tool-version "rocprofv3 7.1.1" --gpu-architecture gfx1200 --workload-limit 1`
  - Result: exit code `0`
  - Summary: `out/rdna4-profiler-backed-timing-smoke/summary.json`
  - `profiler_collected=true`
  - `policy_backend=rocprofv3`
  - `activity_domain=kernel_activity`
  - `csv_path=out/rdna4-profiler-backed-timing-smoke/rocprofv3/rdna4-triton-rmsnorm-timing_kernel_trace.csv`
  - `kernel_duration_ms=5.550257`

## Claim Boundary

This creates bounded RDNA4 profiler-backed timing smoke evidence only. It does
not upgrade RDNA4 to full paper validation, score authority, leaderboard
readiness, broader AMD hardware validation, CDNA3/MI300X validation, or CDNA4
validation.
