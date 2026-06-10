---
phase: 178-derived-script-parallelism
verified: 2026-06-11T00:00:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
gaps: []
---

# Phase 178: Derived Script Parallelism Verification Report

**Phase Goal:** The derived isolation script runs multiple problem subprocesses concurrently, improving throughput for CPU-bound derived sidecar generation without affecting GPU correctness  
**Verified:** 2026-06-11  
**Status:** PASSED  
**Re-verification:** No — initial verification  

## Goal Achievement

### Observable Truths

| #   | Truth   | Status     | Evidence       |
| --- | ------- | ---------- | -------------- |
| 1   | run_derived_isolated.py dispatches per-problem subprocesses concurrently via ThreadPoolExecutor instead of a serial for-loop | ✓ VERIFIED | Line 384: `with ThreadPoolExecutor(max_workers=args.jobs) as executor:`; Line 20: `from concurrent.futures import ThreadPoolExecutor, as_completed` |
| 2   | Status JSONL writes are thread-safe -- concurrent workers never produce interleaved or corrupted lines | ✓ VERIFIED | Line 289: `with status_lock:` protecting `append_status()`; Line 380: `status_lock = threading.Lock()`; test_thread_safe_jsonl_writes passes with 10 threads × 100 lines |
| 3   | Existing --resume and --continue-on-failure semantics produce identical results under parallel execution as under serial execution | ✓ VERIFIED | Line 372: `completed = load_completed(args.status_jsonl) if args.resume else set()`; test_parallel_resume_semantics verifies atomic skipping; all existing tests pass |
| 4   | --jobs flag controls concurrency level with a sensible default (e.g., min(os.cpu_count(), 4)) | ✓ VERIFIED | Lines 352-355: `--jobs` argument with `default=min(os.cpu_count() or 1, 4)`; test_jobs_flag_default passes |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected    | Status | Details |
| -------- | ----------- | ------ | ------- |
| `scripts/run_derived_isolated.py` | ThreadPoolExecutor-based parallel subprocess dispatch | ✓ VERIFIED | 419 lines; contains `from concurrent.futures import ThreadPoolExecutor`; contains `def _partition_problems_by_index`; all helper functions present |
| `tests/sol_execbench/test_run_derived_isolated.py` | Parallel execution tests for DERV-01 through DERV-04 | ✓ VERIFIED | 399 lines; contains `test_thread_safe_jsonl_writes`, `test_parallel_resume_semantics`, `test_jobs_flag_default`, `test_parallel_dispatch`; all 9 tests pass |

### Key Link Verification

| From | To  | Via | Status | Details |
| ---- | --- | --- | ------ | ------- |
| `scripts/run_derived_isolated.py` | `threading.Lock` | `append_status()` function | ✓ WIRED | Line 289: `with status_lock:` protecting JSONL writes |
| `scripts/run_derived_isolated.py` | `concurrent.futures.ThreadPoolExecutor` | `main()` function | ✓ WIRED | Line 384: `with ThreadPoolExecutor(max_workers=args.jobs) as executor:` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| `_process_problem_chunk()` | `problem_id`, `chunk_results` | Worker thread subprocess calls | ✓ FLOWING | Line 283: `status = run_problem()` calls real subprocess; Line 290: `append_status()` writes real data |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| All tests pass | `uv run pytest tests/sol_execbench/test_run_derived_isolated.py -x` | 9/9 passed in 1.99s | ✓ PASS |
| ThreadPoolExecutor pattern verified | AST analysis of script | All critical elements present | ✓ PASS |
| Commits exist | `git log --oneline` | 748b8ae, e9bdad6 present | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| DERV-01 | 178-01-PLAN.md | Parallel subprocess dispatch via ThreadPoolExecutor | ✓ SATISFIED | test_parallel_dispatch passes |
| DERV-02 | 178-01-PLAN.md | Thread-safe JSONL writes | ✓ SATISFIED | test_thread_safe_jsonl_writes passes |
| DERV-03 | 178-01-PLAN.md | Parallel resume semantics | ✓ SATISFIED | test_parallel_resume_semantics passes |
| DERV-04 | 178-01-PLAN.md | --jobs flag with default min(cpu_count, 4) | ✓ SATISFIED | test_jobs_flag_default passes |

### Anti-Patterns Found

No anti-patterns detected. No debt markers (TBD, FIXME, XXX) found. The only `return []` statement is a valid boundary condition check in `_partition_problems_by_index()` (line 251), not a stub.

### Human Verification Required

None. All verification criteria are fully satisfied by automated tests and code inspection.

### Gaps Summary

No gaps found. All must-haves verified successfully:

1. **Thread-safe concurrent execution**: ThreadPoolExecutor implementation follows Phase 177 pattern with pre-partitioned chunks and thread-safe JSONL writes
2. **Resume semantics preserved**: Completed set loaded once before parallel execution, ensuring atomic skip behavior
3. **Concurrency control**: --jobs flag with conservative default prevents resource exhaustion
4. **Error handling preserved**: --continue-on-failure semantics maintained under parallel execution
5. **Deterministic output**: Results sorted by problem_id before final exit code calculation
6. **Interrupt handling**: KeyboardInterrupt returns exit code 130 for structured partial-completion output

The implementation successfully achieves the phase goal: derived sidecar generation now runs multiple problem subprocesses concurrently via ThreadPoolExecutor, improving throughput for CPU-bound operations while maintaining all existing correctness and safety guarantees.

---

_Verified: 2026-06-11_  
_Verifier: Claude (gsd-verifier)_