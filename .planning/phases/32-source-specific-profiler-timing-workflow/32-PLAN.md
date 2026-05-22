# Phase 32 Plan: Source-Specific Profiler Timing Workflow

**Status:** Complete

## Tasks

- [x] Extend ROCm profiler evidence with warmup count, iteration count, trial
  count, clock-lock status, GPU architecture, backend, aggregation rule, and
  fallback reason.
- [x] Add a source-specific collection helper that maps solution languages to
  the appropriate timing policy before invoking `rocprofv3` or returning
  fallback metadata.
- [x] Refactor dataset CLI command construction so normal runs and profiler
  evidence collection use the same benchmark command.
- [x] Add dataset options for opt-in timing evidence collection,
  architecture/tool-version annotation, warmup runs, and clock-lock config.
- [x] Add fixture-backed parser and workflow tests for kernel rows, HIP runtime
  rows, missing CSV output, command failure, Triton profiler collection, and
  PyTorch fallback.
- [x] Document the timing evidence workflow and report fields.

## Verification

- `uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_run_dataset_amd_score.py`
- `uv run ruff check src/sol_execbench/core/bench/rocm_profiler.py scripts/run_dataset.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_run_dataset_amd_score.py`
