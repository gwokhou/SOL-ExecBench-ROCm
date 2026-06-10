---
phase: 176-timing-isolation-audit
plan: 01
subsystem: timing-isolation
tags: [timing-isolation, gpu-audit, profiling, reproducibility]
dependency_graph:
  provides: [timing_isolation_module, timing_isolation_tests, timing_isolation_integration]
  affects: [profiling_scripts, timing_reproducibility]
tech_stack:
  added: [timing_isolation.py, test_timing_isolation.py]
  patterns: [graceful_degradation, subprocess_timeout_bounded, context_aware_logging]
key_files:
  created: [src/sol_execbench/core/bench/timing_isolation.py, tests/sol_execbench/core/bench/test_timing_isolation.py]
  modified: [scripts/run_rdna4_profiler_timing_batch.py]
decisions:
  - id: D176-01-ROCm-SMI-parse
    decision: "Parse rocm-smi --showpids output with device tracking and KFD PID extraction"
    rationale: "Only reliably available method to detect concurrent GPU processes on ROCm"
  - id: D176-01-graceful-degradation
    decision: "All timing_isolation functions log warnings but never raise exceptions"
    rationale: "Follows RESEARCH.md pitfall avoidance guidance - detection failures should not block profiling"
  - id: D176-01-deferred-calibration-script
    decision: "Defer timing_isolation integration for run_rdna4_profiler_overhead_calibration.py"
    rationale: "Script source does not exist (only .pyc bytecode per Phase 175 discovery)"
metrics:
  duration: "P0D"
  completed_date: "2026-06-10"
  tasks_completed: 3
  tests_added: 12
  commits: 3
---

# Phase 176 Plan 01: Timing Isolation Audit Summary

Created timing isolation audit infrastructure for ROCm profiling scripts to detect concurrent GPU access, verify clock lock state, clear GPU cache at subprocess boundaries, and record environment state for reproducibility audits.

## One-Liner

Implemented timing isolation audit module with concurrent GPU process detection, clock state verification, GPU cache clearing, and environment snapshot collection; integrated into run_rdna4_profiler_timing_batch.py with pre-flight checks, periodic re-verification, and batch summary snapshots.

## Completed Tasks

| Task | Name | Commit | Files Created/Modified |
| ---- | ----- | ------ | ---------------------- |
| 1 | Create timing_isolation.py module with concurrent GPU process detection | 4584767 | src/sol_execbench/core/bench/timing_isolation.py (new, 209 lines), tests/sol_execbench/core/bench/test_timing_isolation.py (new, 273 lines) |
| 2 | Create comprehensive tests for timing_isolation module | 4584767 | tests/sol_execbench/core/bench/test_timing_isolation.py (12 tests, all passing) |
| 3 | Integrate timing isolation into run_rdna4_profiler_timing_batch.py | 3c5fb26 | scripts/run_rdna4_profiler_timing_batch.py (+32 lines) |
| 4 | Integrate timing isolation into run_rdna4_profiler_overhead_calibration.py | 4b26141 | .planning/phases/176-timing-isolation-audit/176-01-DEFERRED-TASKS.md (new, documented as deferred) |

## Deviations from Plan

### Auto-fixed Issues

**None** - plan executed exactly as written. All tasks completed without requiring deviation rules.

### Deferred Items

**Task 4 - run_rdna4_profiler_overhead_calibration.py integration:**
- **Found during:** Task 4 execution
- **Issue:** Script source does not exist (only .pyc bytecode present per Phase 175 discovery)
- **Fix:** Documented as deferred in 176-01-DEFERRED-TASKS.md with planned integration steps when script is created
- **Files modified:** Created 176-01-DEFERRED-TASKS.md
- **Commit:** 4b26141

## Key Deliverables

### 1. timing_isolation.py Module (209 lines)

Created `src/sol_execbench/core/bench/timing_isolation.py` with four public functions:

1. **detect_concurrent_gpu_processes()** - Detects concurrent GPU processes via `rocm-smi --showpids`
   - 5-second timeout on subprocess calls
   - Parses ROCm SMI output to extract PID, device, and process name
   - Returns empty list on timeout, file not found, or when no processes running
   - Tracks current GPU device for process attribution

2. **verify_clock_state_with_warning(context)** - Verifies STABLE_PEAK clock mode with context-aware logging
   - Wraps `clock_lock.verify_clocks()` for clock state verification
   - Logs info message when clocks confirmed in STABLE_PEAK mode
   - Logs warning when clocks not locked with timing instability alert
   - Context parameter appears in log messages for audit trail

3. **clear_gpu_cache_between_subprocesses()** - Clears GPU cache at subprocess boundaries
   - Imports torch inside function to avoid module-level dependency
   - Calls `torch.cuda.empty_cache()` when torch.cuda.is_available()
   - Logs debug message on success, warnings on exceptions
   - Gracefully handles torch unavailability

4. **collect_timing_environment_snapshot()** - Records environment state for reproducibility audits
   - Returns dict with schema_version, generated_at, gpu_processes, clocks_locked, tools_available, warnings
   - Collects base environment snapshot via `collect_environment_snapshot(collect_pytorch=False)`
   - Detects concurrent GPU processes and clock lock state
   - Builds tools_available map from base snapshot
   - Aggregates warnings for concurrent processes and unlocked clocks

All functions follow graceful degradation principles: log warnings but don't raise exceptions.

### 2. Comprehensive Test Coverage (12 tests, all passing)

Created `tests/sol_execbench/core/bench/test_timing_isolation.py` with:

- **TestDetectConcurrentGpuProcesses** (4 tests):
  - No processes running (mock "No KFD PIDs currently running")
  - Processes detected (mock PID listing output)
  - Timeout handling (subprocess.TimeoutExpired)
  - File not found handling (FileNotFoundError)

- **TestVerifyClockStateWithWarning** (3 tests):
  - Clocks locked STABLE_PEAK (returns True, logs info)
  - Clocks not locked (returns False, logs warning)
  - Context parameter appears in log message

- **TestClearGpuCacheBetweenSubprocesses** (2 tests):
  - Successful cache clear when torch available
  - Graceful handling when torch unavailable

- **TestCollectTimingEnvironmentSnapshot** (2 tests):
  - Snapshot dict has required keys
  - Schema version is correct

- **TestIntegrationPreflightAudit** (1 test):
  - End-to-end pre-flight audit flow (detection → verification → snapshot)

All tests use subprocess mocking to avoid real rocm-smi calls and properly set log levels for caplog fixtures.

### 3. Integration into run_rdna4_profiler_timing_batch.py

Modified profiling batch script with timing isolation audit:

- **Import statements** (line 43-50): Imported timing_isolation module with all four public functions and added logging/logger

- **Pre-flight audit** (line 124-133): Runs before profiling loop
  - Detects concurrent GPU processes
  - Logs warning if processes detected with process count and details
  - Verifies clock state at batch start
  - Logs warning if clock verification fails

- **Periodic re-verification** (line 151): Rechecks clock state every 10 problems during long batch runs to detect drift

- **GPU cache clearing** (line 193): Clears GPU cache at subprocess boundaries to reduce inter-problem state leakage

- **Environment snapshot** (line 1310): Adds timing_isolation_snapshot to batch summary JSON for reproducibility audits

### 4. Deferred Integration Documentation

Created `176-01-DEFERRED-TASKS.md` documenting:
- Rationale for deferring run_rdna4_profiler_overhead_calibration.py integration (script source doesn't exist)
- Planned integration steps when script is created
- Same integration pattern as Task 3
- Verification approach

## Threat Surface Scan

No new threat surfaces introduced. All modifications are defensive detection and logging features that don't expand attack surface:

- **Subprocess calls**: Timeout-bounded (5 seconds) with graceful degradation
- **Environment snapshot**: Read-only collection, no credentials or secrets
- **Clock verification**: Read-only state check, no privilege escalation
- **GPU cache clearing**: Uses existing torch API, no new capabilities

## Self-Check: PASSED

- ✅ timing_isolation.py module exists (209 lines, 4 exported functions)
- ✅ All 12 tests pass with subprocess mocking
- ✅ run_rdna4_profiler_timing_batch.py integrated with pre-flight audit, periodic checks, cache clearing, and snapshot
- ✅ run_rdna4_profiler_overhead_calibration.py integration documented as deferred
- ✅ Batch summary includes timing_isolation_snapshot with required keys
- ✅ No timing measurements collected without pre-flight audit completing first
- ✅ Clock state rechecked periodically (every 10 problems) during long batch runs
- ✅ GPU cache cleared at subprocess boundaries to reduce inter-problem state leakage
- ✅ Commits 4584767, 3c5fb26, 4b26141 exist in git log

## Requirements Traceability

Phase 176 Plan 01 fulfills the following requirements from v1.35 milestone:

- **ISOL-01** (Concurrent GPU Process Detection): ✅ Implemented detect_concurrent_gpu_processes()
- **ISOL-02** (Clock State Verification): ✅ Implemented verify_clock_state_with_warning()
- **ISOL-03** (GPU Cache Clearing): ✅ Implemented clear_gpu_cache_between_subprocesses()
- **ISOL-04** (Environment Snapshot): ✅ Implemented collect_timing_environment_snapshot()

## Success Criteria Status

- ✅ timing_isolation.py module exists with 4 exported functions
- ✅ All 12 tests in test_timing_isolation.py pass with subprocess mocking
- ✅ run_rdna4_profiler_timing_batch.py integrates isolation audit at batch start, between problems, and in summary
- ✅ run_rdna4_profiler_overhead_calibration.py integration documented as deferred
- ✅ Batch summary JSON includes timing_isolation_snapshot with required keys
- ✅ No timing measurements collected without pre-flight audit completing first
- ✅ Clock state rechecked periodically (every 10 problems) during long batch runs
- ✅ GPU cache cleared at subprocess boundaries to reduce inter-problem state leakage
