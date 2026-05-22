# Phase 32 Summary: Source-Specific Profiler Timing Workflow

**Completed:** 2026-05-22
**Status:** Complete
**Requirements:** TIME-01, TIME-02, TIME-03, TIME-04

## What Changed

- Added source-specific profiler collection via
  `collect_source_timing_evidence()`.
- Extended `Rocprofv3TimingEvidence` with run metadata: warmup runs,
  iterations, trial count, clock-lock status, tool version, GPU architecture,
  backend, aggregation rule, and fallback reason.
- Added `scripts/run_dataset.py --timing-evidence-dir` with optional
  `--gpu-architecture`, `--timing-tool-version`, `--warmup-runs`, and
  `--lock-clocks` integration.
- Refactored dataset benchmark command construction so regular execution and
  profiler collection share the same command shape.
- Added fixture-backed tests for kernel/HIP runtime parsing, missing-output and
  command-failure fallbacks, source-specific Triton collection, and PyTorch
  fallback routing.
- Updated analysis documentation with the timing evidence workflow.

## Verification

- `uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_run_dataset_amd_score.py` - passed
- `uv run ruff check src/sol_execbench/core/bench/rocm_profiler.py scripts/run_dataset.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_run_dataset_amd_score.py` - passed

## Compatibility

Canonical trace JSONL was not changed. Timing evidence is an optional derived
artifact written by the dataset workflow when requested.
