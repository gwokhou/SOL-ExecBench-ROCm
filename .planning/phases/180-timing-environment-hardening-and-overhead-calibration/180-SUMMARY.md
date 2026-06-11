---
phase: 180-timing-environment-hardening-and-overhead-calibration
plan: 01+02
type: execute
subsystem: timing-isolation, profiler-timing-batch-script, rocm-profiler
tags: [gpu-isolation, strict-isolation, rocr-visible-devices, overhead-calibration, rocprofv3]
key_files_created:
  - scripts/run_rdna4_profiler_overhead_calibration.py
  - tests/sol_execbench/test_rdna4_profiler_overhead_calibration.py
key_files_modified:
  - src/sol_execbench/core/bench/timing_isolation.py
  - src/sol_execbench/core/bench/rocm_profiler.py
  - scripts/run_rdna4_profiler_timing_batch.py
  - tests/sol_execbench/core/bench/test_timing_isolation.py
decisions:
  - "validate_gpu_device_isolation() returns structured result; caller decides warn vs abort"
  - "--strict-isolation defaults to False on batch script, True on calibration script"
  - "Overhead calibration uses inner subprocess under rocprofv3 for consistent measurement"
  - "profiler_overhead_ms added as optional field to Rocprofv3TimingEvidence (backward compatible)"
  - "Calibration path read via _read_overhead_calibration() with graceful degradation"
metrics:
  duration: ~45 minutes
  tasks_completed: 6 of 6
  test_pass_rate: 1582/1582 (15 new tests)
  lines_added: ~450
  lines_removed: ~5
---

# Phase 180: Timing Environment Hardening and Overhead Calibration Summary

## Objective

Close three remaining gaps in timing data precision: GPU device isolation, strict isolation
abort mode, and rocprofv3 overhead calibration.

## Implementation

### Plan 180-01: GPU Device Isolation and Strict Isolation Mode

**Task 1: `validate_gpu_device_isolation()` in timing_isolation.py**

Added `_detect_gpu_count()` helper (rocm-smi --showid, 5s timeout, graceful degradation) and
`validate_gpu_device_isolation()` public function that:
- Optionally sets `ROCR_VISIBLE_DEVICES` when `gpu_device` provided
- Checks GPU count vs device restriction status
- Returns structured result: `{isolated, gpu_count, rocr_visible_devices, gpu_device_set, warnings}`

Updated `collect_timing_environment_snapshot()` to include `gpu_isolation` field.

**Task 2: `--strict-isolation` and `--gpu-device` flags in batch script**

Added two CLI arguments to `run_rdna4_profiler_timing_batch.py`:
- `--strict-isolation` (bool, default False)
- `--gpu-device` (int, optional)

**Task 3: Strict isolation abort logic**

In `run_batch()` pre-flight audit:
- Concurrent GPU processes → warn (default) or abort (strict)
- Clock state not STABLE_PEAK → warn (default) or abort (strict)
- GPU device not isolated → warn (default) or abort (strict)

**Task 4: Tests**

- 5 tests for `validate_gpu_device_isolation()` (single GPU, multi-GPU, device set, unknown)
- 6 tests for batch script flags and strict isolation behavior

### Plan 180-02: rocprofv3 Overhead Calibration

**Task 1: Calibration script**

Created `scripts/run_rdna4_profiler_overhead_calibration.py`:
- Runs vector add kernel (1M elements) without rocprofv3 → baseline
- Runs same kernel under rocprofv3 → instrumented
- Computes overhead = profiler_median - baseline_median
- Writes versioned JSON calibration sidecar
- Integrates PID lock (mandatory), timing isolation audit, --strict-isolation (default True)

**Task 2: profiler_overhead_ms in timing evidence**

Extended `Rocprofv3TimingEvidence` with `profiler_overhead_ms: float | None` field.
Extended `build_timing_evidence()` to accept `profiler_overhead_ms` parameter.
Added `_read_overhead_calibration()` helper with graceful degradation.
Extended `collect_rocprofv3_timing()` to accept `calibration_path` parameter.

**Task 3: Tests**

- 5 tests for `profiler_overhead_ms` in evidence model and build function
- 5 tests for `_read_overhead_calibration()` (valid, missing, invalid JSON, missing key)
- 4 tests for calibration script argument parsing
- 1 test for calibration schema version

## Requirements Coverage

- **ENV-01** ✅: GPU device isolation validated via `validate_gpu_device_isolation()`
- **ENV-02** ✅: `--strict-isolation` abort mode implemented for batch and calibration scripts
- **ENV-03** ✅: Overhead calibration script created with versioned JSON output
- **ENV-04** ✅: `profiler_overhead_ms` integrated into timing evidence chain

## Verification

- 1582/1582 tests passing (15 new)
- 0 regressions
- ROCm migration residue audit passing
