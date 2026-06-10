---
phase: 178-derived-script-parallelism
plan: 01
subsystem: derived-sidecar-generation
tags: [parallelism, threadpoolexecutor, derived-sidecars, script-refactoring]
wave: 1
dependency_graph:
  requires: []
  provides: [parallel-derived-execution]
  affects: [run_derived_isolated.py]
tech_stack:
  added: [concurrent.futures.ThreadPoolExecutor, threading.Lock]
  patterns: [pre-partitioning, worker-chunks, thread-safe-file-writes]
key_files:
  created: []
  modified: [scripts/run_derived_isolated.py, tests/sol_execbench/test_run_derived_isolated.py]
decisions: []
metrics:
  duration: "PT15M"
  completed_date: "2026-06-11"
---

# Phase 178 Plan 01: Derived Script Parallelism Summary

Successfully implemented ThreadPoolExecutor-based concurrent subprocess dispatch in `run_derived_isolated.py`, following the proven Phase 177 pattern for parallel batch processing. All four requirements (DERV-01 through DERV-04) have been implemented and verified with comprehensive test coverage.

## One-Liner

Refactored `run_derived_isolated.py` to use ThreadPoolExecutor for concurrent per-problem derived sidecar generation with thread-safe JSONL writes, preserving existing resume semantics and failure handling.

## Implementation Summary

### Task 0: Test Infrastructure
Created comprehensive test infrastructure covering all four Phase 178 requirements:

- **test_thread_safe_jsonl_writes** (DERV-02): Verifies concurrent `append_status()` calls from 10 threads writing 100 lines each produce no interleaved or corrupted JSONL lines
- **test_parallel_resume_semantics** (DERV-03): Confirms parallel workers skip completed problems atomically when `--resume` flag is set
- **test_jobs_flag_default** (DERV-04): Validates `--jobs` defaults to `min(os.cpu_count(), 4)` when not specified
- **test_parallel_dispatch** (DERV-01): Integration test with 4 problems and `--jobs=2` verifying all problems complete and status JSONL contains all entries

### Task 1: ThreadPoolExecutor Implementation
Refactored `run_derived_isolated.py` main execution loop following Phase 177 pattern:

- **Imports**: Added `threading` and `concurrent.futures.ThreadPoolExecutor, as_completed`
- **CLI argument**: Added `--jobs` with default `min(os.cpu_count(), 4)` and argparse validation
- **Helper functions**:
  - `_partition_problems_by_index()`: Pre-partitions problems into exclusive chunks for workers
  - `_process_problem_chunk()`: Processes a chunk serially within a worker thread with status_lock protection
- **main() refactoring**:
  - Replaced serial for-loop with ThreadPoolExecutor context manager
  - Pre-partitions problems before spawning workers
  - Collects results via `as_completed()` for robust error handling
  - Sorts results by `problem_id` for deterministic output
  - Handles KeyboardInterrupt with exit code 130
- **Thread safety**: `append_status()` calls protected by `threading.Lock` via `with status_lock:`

### Task 2: Verification
All verification checks passed:

- ✅ Thread-safe JSONL writes verified via concurrent write test (10 threads × 100 lines = 1000 lines, all valid JSON)
- ✅ Parallel resume semantics verified (completed problems skipped atomically, no duplicate entries)
- ✅ `--jobs` flag working (default tested, explicit `--jobs=1` serial and `--jobs=4` parallel both work)
- ✅ Parallel dispatch verified (4 problems with 2 workers complete correctly)
- ✅ Output determinism verified (results sorted by problem_id regardless of completion order)
- ✅ KeyboardInterrupt handling produces exit code 130
- ✅ All existing tests continue passing

## Deviations from Plan

**None** - Plan executed exactly as written.

## Auth Gates

**None** - No authentication gates encountered during this phase.

## Threat Surface Scan

No new security-relevant surface introduced beyond existing subprocess execution:

- **Existing**: Script already executes subprocesses via `subprocess.run()` - unchanged
- **Added**: Thread-safe file writes via `threading.Lock` - standard stdlib, no external deps
- **Added**: `--jobs` argument validation via argparse - caps at reasonable max (default min(cpu_count, 4))
- **Risk**: Concurrent subprocess execution increases resource utilization but does not introduce new attack vectors

All mitigations from threat model implemented:
- ✅ T-178-01: `--jobs` argument validated by argparse (positive integer, conservative default)
- ✅ T-178-02: `--status-jsonl` path validated by argparse `type=Path`
- ✅ T-178-03: `load_problem_id_filter()` strips comments/whitespace, validates format
- ✅ T-178-05: Python stdlib only (threading, concurrent.futures) - no new dependencies

## Known Stubs

**None** - All functionality implemented and tested. No placeholder code or TODOs remain.

## Self-Check: PASSED

- [x] Modified files exist: `scripts/run_derived_isolated.py`, `tests/sol_execbench/test_run_derived_isolated.py`
- [x] Commits exist:
  - `748b8ae`: test(178-01): add parallel execution test infrastructure
  - `e9bdad6`: feat(178-01): add ThreadPoolExecutor parallelism to run_derived_isolated.py
- [x] All tests passing (9/9)
- [x] SUMMARY.md created in plan directory

## Requirements Traceability

All plan requirements implemented and verified:

| Requirement | Description | Verification |
|------------|-------------|--------------|
| DERV-01 | Parallel subprocess dispatch via ThreadPoolExecutor | test_parallel_dispatch |
| DERV-02 | Thread-safe JSONL writes | test_thread_safe_jsonl_writes |
| DERV-03 | Parallel resume semantics | test_parallel_resume_semantics |
| DERV-04 | --jobs flag with default min(cpu_count, 4) | test_jobs_flag_default |

## Success Criteria

All success criteria met:

- [x] run_derived_isolated.py uses ThreadPoolExecutor for concurrent subprocess dispatch
- [x] Status JSONL writes are thread-safe (verified via concurrent write test)
- [x] --resume and --continue-on-failure produce identical results under parallel execution
- [x] --jobs flag controls concurrency with sensible default (min(os.cpu_count(), 4))
- [x] All existing tests pass + 4 new tests for parallel execution
- [x] KeyboardInterrupt handling produces structured partial-completion output (exit code 130)
- [x] Output order is deterministic (sorted by problem_id)

## Next Steps

Phase 178 Plan 01 is complete. The derived sidecar generation script now supports concurrent execution, reducing end-to-end runtime for CPU-bound derived sidecar generation while preserving all existing semantics and failure handling.
