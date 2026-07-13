# Phase 28 Summary: Live rocprofv3 Timing Integration

**Completed:** 2026-05-22
**Status:** Complete
**Requirements:** PROF-01, PROF-02, PROF-03, PROF-04

## What Changed

- Added `Rocprofv3CollectionRequest` and `Rocprofv3CollectionResult` derived
  models.
- Added `collect_rocprofv3_timing()` with injectable runner support so live
  profiler collection can be tested without real GPU/profiler access.
- Preserved source-specific policy routing: HIP native and Triton can collect
  `rocprofv3` kernel activity; PyTorch/mixed/unknown paths return explicit
  fallback selection metadata instead of pretending to be kernel activity.
- Added fallback labeling for profiler command failures and missing CSV output.
- Documented the live adapter and chimney-style timing semantics in
  `docs/user/rocm_timing.md`.

## Verification

- `uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py` - passed
- `uv run ruff check src/sol_execbench/core/bench/rocm_profiler.py tests/sol_execbench/test_rocm_profiler.py` - passed

## Compatibility

`time_runnable()` and canonical trace JSONL were not changed. Live timing
evidence remains a derived artifact.
