# Phase 180: Timing Environment Hardening and Overhead Calibration

## Problem Statement

Three gaps remain in the benchmark pipeline that compromise timing data precision:

1. **No ROCR_VISIBLE_DEVICES enforcement**: Scripts do not set or validate GPU device
   visibility. Other processes (training scripts, notebooks) can silently share the GPU,
   introducing timing bias that PID locking alone cannot prevent.

2. **Concurrent process detection is warn-only**: `detect_concurrent_gpu_processes()`
   (Phase 176) logs warnings but never aborts. For timing-sensitive measurements
   (profiler-backed timing collection), detected contamination silently pollutes data.
   A `--strict-isolation` mode is needed for data-measurement-sensitive scenarios.

3. **rocprofv3 overhead uncalibrated**: The profiler-backed timing pipeline
   (`collect_rocprofv3_timing`) measures kernel durations through rocprofv3
   instrumentation, but the instrumentation overhead itself has never been quantified.
   Without a known overhead baseline, profiler-backed timing lacks absolute accuracy
   claims. The overhead calibration script was planned in the original v1.35 milestone
   (Phase 175-01) but was rolled back and never rebuilt.

## Existing Infrastructure

- `timing_isolation.py` (Phase 176): concurrent GPU process detection, clock verification,
  cache clearing, environment snapshots — all warn-only, never abort
- `pid_lock.py` (Phase 175): fcntl.flock-based exclusive lock — prevents multi-instance
  of the *same* script but not interference from unrelated processes
- `rocm_profiler.py`: `collect_rocprofv3_timing()` — profiler-backed timing collection
  without overhead compensation
- `runtime_evidence.py`: already collects `ROCR_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`
  for audit but does not enforce them
- Phase 176-01-DEFERRED-TASKS.md: documents deferred integration of timing isolation into
  the overhead calibration script (script did not exist)

## Scope

### In Scope

- Add GPU device isolation enforcement to timing-sensitive scripts
- Add `--strict-isolation` mode to `run_rdna4_profiler_timing_batch.py` and future
  timing-sensitive scripts
- Create `run_rdna4_profiler_overhead_calibration.py` script to measure and record
  rocprofv3 instrumentation overhead
- Integrate overhead calibration results into timing evidence metadata
- Update evaluation stability reason codes for new failure modes

### Out of Scope

- Temperature/power monitoring (separate concern, deferred)
- Cooldown periods between problems (separate concern, deferred)
- Fixing Phase 177 mock-incompatible tests (existing known issue, not related)

## Dependencies

- Phase 175 (PID lock module) — overhead calibration script uses `acquire_pid_lock()`
- Phase 176 (timing isolation audit) — `--strict-isolation` extends this module
- Phase 177 (profiler timing batch parallelism) — `--strict-isolation` flag added here
- Phase 179 (evaluation stability extension) — new reason codes integrate here

## Requirements

### ENV-01: GPU Device Isolation

Timing-sensitive scripts MUST validate that `ROCR_VISIBLE_DEVICES` is set or that only
one GPU is visible. When multiple GPUs are visible and no device restriction is in place,
the script MUST warn in default mode and abort in `--strict-isolation` mode.

A new `--gpu-device` flag allows explicit device targeting (sets `ROCR_VISIBLE_DEVICES`
for the current process).

### ENV-02: Strict Isolation Mode

A `--strict-isolation` flag for `run_rdna4_profiler_timing_batch.py` upgrades all
graceful-degradation warnings to hard failures:

- Concurrent GPU processes detected → abort (exit 1) instead of warn
- Clock not in STABLE_PEAK → abort instead of warn
- ROCR_VISIBLE_DEVICES not set with multi-GPU system → abort instead of warn

The flag defaults to False (current behavior preserved).

### ENV-03: rocprofv3 Overhead Calibration

A new script `scripts/run_rdna4_profiler_overhead_calibration.py` measures rocprofv3
instrumentation overhead:

1. Runs a minimal HIP kernel (e.g., vector add) N times WITHOUT rocprofv3 → baseline
2. Runs the same kernel N times WITH rocprofv3 → instrumented timing
3. Computes overhead = instrumented_median - baseline_median
4. Writes calibration result to a JSON sidecar with schema versioning

The calibration value is recorded in batch summary metadata so downstream consumers
can account for profiler overhead in absolute timing claims.

### ENV-04: Overhead Integration

`collect_rocprofv3_timing()` results include a `profiler_overhead_ms` field when a
calibration file is available. Evaluation stability reports reference this value in
their `profiler_overhead_risk` reason code logic.
