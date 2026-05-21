---
phase: 03-rocm-evaluation-timing-and-hardware-introspection
plan: 02
subsystem: timing
tags:
  - rocm
  - timing
key-files:
  - src/sol_execbench/core/bench/timing.py
  - tests/sol_execbench/core/bench/test_timing.py
metrics:
  tests: "57 skipped in current non-GPU marker configuration"
---

# Plan 03-02 Summary

## Changes

- Removed the import-time CUPTI dependency from `src/sol_execbench/core/bench/timing.py`.
- Made `time_runnable()` default to ROCm-compatible PyTorch device event timing.
- Kept compatibility wrappers for existing `bench_time_with_cuda_events()` and `bench_gpu_time_with_cupti()` callers, with the latter routed to device events instead of CUPTI.
- Preserved shifting allocator setup exclusion, cache clearing, warmup/rep handling, and return modes.
- Added a post-call device synchronization before recording the end event so submitted non-default-stream work is not hidden from the timing interval.

## Verification

- `uv run --no-sync pytest tests/sol_execbench/core/bench/test_timing.py` -> 57 skipped under the repository's current `timing_serial` marker configuration.
- `uv run --no-sync pytest tests/sol_execbench/core/bench/test_timing.py tests/sol_execbench/test_rocm_eval_timing_audit.py` -> 3 passed, 57 skipped.

## Deviations

- Existing timing tests are marker-skipped in this environment, so Plan 03-04 will add a non-GPU source audit asserting Phase 3 no longer imports or calls CUPTI.

## Self-Check: PASSED

The default runtime timing path no longer depends on CUPTI and remains API-compatible for existing callers.
