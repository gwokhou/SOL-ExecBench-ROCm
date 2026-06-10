# Phase 179: Evaluation Stability Extension and Integration Tests - Research

**Researched:** 2026-06-11
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase)

## Domain Analysis

Phase 179 extends the evaluation stability diagnostics with new concurrency-related failure mode detection and creates integration tests that verify the complete parallelism and safety hardening system implemented across Phases 175-178.

### New Reason Codes

The `evaluation_stability.py` module currently has these reason codes:
- `backend_unsupported`
- `missing_timing`
- `insufficient_samples`
- `clock_unlocked`
- `profiler_overhead_risk`
- `noisy`
- `stable`

Two new reason codes are needed:

1. **`gpu_contention`**: Indicates concurrent GPU processes were detected during timing measurement, suggesting the timing may be inflated due to GPU resource contention. This would be set by the `timing_isolation.py` module's `detect_concurrent_gpu_processes()` function when concurrent processes are found.

2. **`multi_instance_interference`**: Indicates multiple evaluation instances were running concurrently (detected via PID lock contention), suggesting potential multi-instance interference. This would be set when the PID lock module detects another instance holding the lock.

### Integration Test Requirements

Integration tests must verify:

1. **PID Lock Contention**: When a second instance attempts to run while the first holds the lock, the second exits with a clear diagnostic message.

2. **Parallel Staging with Serial Profiling**: The ThreadPoolExecutor-based parallel staging in `run_rdna4_profiler_timing_batch.py` runs CPU work in parallel but keeps GPU profiling strictly serial, ensuring no concurrent GPU subprocess execution.

3. **Isolation Audit Output**: The environment snapshot in the batch summary sidecar includes GPU processes, clock state, and lock status, enabling reproducibility audits.

## Existing Code Context

### Evaluation Stability Module

Located at `src/sol_execbench/core/evaluation_stability.py`:
- Defines `STABILITY_STATUS_KEYS` tuple with existing status values
- Defines `STABILITY_STATUS_PRIORITY` tuple for status ordering
- `StabilityStatusTotals` class has fields for each status type
- `_reason_codes()` function builds the list of reason codes from timing evidence
- Reason codes are added to the list based on detected conditions

### Timing Isolation Module

Located at `src/sol_execbench/core/bench/timing_isolation.py` (created in Phase 176):
- `detect_concurrent_gpu_processes()` function detects concurrent GPU processes via `rocm-smi --showpids`
- `verify_clock_state_with_warning()` function verifies STABLE_PEAK clock mode
- `clear_gpu_cache_between_subprocesses()` function clears GPU cache
- `collect_timing_environment_snapshot()` function records environment state

### PID Lock Module

Located at `src/sol_execbench/core/bench/pid_lock.py` (created in Phase 175):
- `acquire_pid_lock()` context manager acquires exclusive lock via `fcntl.flock`
- Exits with diagnostic message when another instance holds the lock

### Integration Test Patterns

From `tests/sol_execbench/test_e2e.py`:
- Uses `subprocess.run()` for script execution
- Uses `pytest.mark.xdist_group("serial")` for serial test execution
- Verifies exit codes, stdout/stderr, and output file existence
- Loads and validates JSON output

## Implementation Strategy

### Task 1: Add New Reason Codes to Evaluation Stability

1. Add `gpu_contention` and `multi_instance_interference` to `STABILITY_STATUS_KEYS` tuple
2. Add corresponding fields to `StabilityStatusTotals` class
3. Update `_reason_codes()` function to detect these conditions:
   - `gpu_contention`: Check if timing evidence has concurrent GPU processes
   - `multi_instance_interference`: Check if PID lock detected contention
4. Add the new codes to appropriate positions in `STABILITY_STATUS_PRIORITY` tuple

### Task 2: Create Integration Tests

Create `tests/sol_execbench/test_v1_35_integration.py` with three test classes:

1. **TestPidLockContention**: Verify second instance is rejected when first holds lock
   - Start first instance with long-running operation
   - Attempt to start second instance concurrently
   - Verify second instance exits with non-zero code and diagnostic message
   - Verify first instance completes successfully

2. **TestParallelStagingSerialProfiling**: Verify GPU profiling remains serial
   - Mock GPU profiling calls to track execution order
   - Run batch with multiple targets and max_workers=4
   - Verify CPU staging operations run concurrently
   - Verify GPU profiling calls are strictly sequential

3. **TestIsolationAuditOutput**: Verify environment snapshot is well-formed
   - Run timing batch with timing isolation enabled
   - Load batch summary JSON
   - Verify timing_isolation_snapshot exists with required keys
   - Verify snapshot contains GPU processes, clock state, lock status

## Standard Stack

No new libraries needed. Phase uses:
- `pytest` for testing (existing)
- `pydantic` for data validation (existing)
- `subprocess` for script execution (existing)
- `fcntl` for file locking (existing)

## Architecture Patterns

Follows established patterns from Phases 175-178:
- Graceful degradation (warnings, not exceptions)
- Context managers for resource management
- Thread-safe operations for parallel execution
- Subprocess-based script isolation
- JSON sidecar files for reproducibility

## Common Pitfalls

1. **Reason Code Ordering**: New reason codes must be added to `STABILITY_STATUS_PRIORITY` in the correct order of severity
2. **Test Flakiness**: Integration tests with real subprocess execution need proper cleanup and timing controls
3. **Mock Complexity**: Testing parallel execution requires careful mock setup to avoid ThreadPoolExecutor compatibility issues
4. **Environment Dependencies**: Tests should work without ROCm hardware (use mocks for rocm-smi)

## Key Implementation Constraints

1. Must maintain backward compatibility with existing stability status codes
2. Must not break existing evaluation stability reports
3. Integration tests should not require actual ROCm GPU hardware
4. New reason codes should follow existing naming conventions

## Success Criteria

1. New reason codes appear in stability diagnostics when appropriate conditions are detected
2. Integration tests pass consistently and verify end-to-end behavior
3. No existing tests break due to reason code additions
4. Documentation is updated with new reason code meanings

## Threat Model

No new security threats introduced. This is purely defensive detection and testing infrastructure:
- Reason codes are read-only diagnostic information
- Integration tests run in isolated subprocess environments
- No privilege escalation or credential handling
- No network operations

## Next Steps

Proceed with planning to create PLAN.md with two main tasks:
1. Extend evaluation stability module with new reason codes
2. Create comprehensive integration tests for the parallelism and safety hardening system
