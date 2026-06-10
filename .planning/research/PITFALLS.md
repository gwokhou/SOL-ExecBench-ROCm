# Domain Pitfalls: Script Parallelism and Safety Hardening

**Domain:** Adding internal CPU-parallel/GPU-serial concurrency, PID locking, and timing reproducibility hardening to existing ROCm GPU benchmark subprocess scripts.
**Researched:** 2026-06-10
**Confidence:** HIGH (patterns derived from code analysis of existing scripts and well-established Python concurrency/file-locking engineering knowledge)

## Critical Pitfalls

Mistakes that cause rewrites, silent data corruption, or timing evidence invalidation.

### Pitfall 1: GPU Subprocess Concurrency Destroys Timing Validity

**What goes wrong:** Running two or more GPU subprocess calls concurrently (e.g., two `rocprofv3` collections or `eval_driver.py` invocations sharing the same GPU) introduces GPU resource contention, L2 cache pollution, and memory bus interference that makes timing measurements meaningless. The entire point of profiler-backed timing replacement is invalidated.

**Why it happens:** `ThreadPoolExecutor` or `ProcessPoolExecutor` with `max_workers > 1` naively applied to the profiling loop in `run_batch()`. The CPU-bound staging work (JSON parsing, `ProblemPackager` construction, workqueue selection) looks parallelizable, and the GPU subprocess call looks like "just another blocking call," so it is tempting to wrap the entire `_profile_target` call in a pool.

**Consequences:** Timing sidecars report numbers that include contention artifacts. These corrupt downstream coverage reports, score reports, and derived evidence. The contamination is silent -- nothing crashes, nothing raises an exception, but the numbers are wrong by potentially large margins (10-50%+ variance on kernel activity duration).

**Prevention:** Enforce GPU-serial execution explicitly. The architecture must be "CPU-parallel staging, GPU-serial profiling." The staging work (definition loading, workload slicing, packager construction, manifest creation) can run in a thread pool, but the actual `subprocess.run()` or `collect_rocprofv3_timing()` call must be serialized through a dedicated GPU mutex or single-consumer queue. Document the GPU-serial constraint in the function signatures and docstrings.

**Detection:** Run the same problem twice: once with internal parallelism, once without. If kernel activity durations differ by more than the established warmup/repeat variance threshold, the parallelism is contaminating GPU timing.

**Phase:** Phase implementing internal parallelism for `run_rdna4_profiler_timing_batch.py`.

### Pitfall 2: File-Based Resume Deduplication Race Condition

**What goes wrong:** The existing `--resume` path checks `_is_classified_replacement_sidecar(replacement_path)` (in `select_fallback_targets`) and `load_completed(status_path)` (in `run_derived_isolated`) to skip already-completed work. Under parallel access, two workers can both observe a target as "not yet completed" and both begin processing it. One writes its result, then the other overwrites it with a different (possibly failed) result.

**Why it happens:** The check-then-act sequence is not atomic. Between `_is_classified_replacement_sidecar` returning `False` and the replacement sidecar being written, another worker performs the same check and also proceeds. This is the classic TOCTOU (time-of-check-time-of-use) race.

**Consequences:** A successfully profiled sidecar can be overwritten by a failed or partial result from a slower worker. The batch summary then reports the problem as failed despite having had a good result. Resume on the next run picks it up again as a target, wasting GPU time. In the worst case, partial_profiler_backed status is downgraded to profiler_blocked.

**Prevention:** Use atomic file creation as the deduplication mechanism. Before starting work on a target, atomically create a lock file or claim marker (e.g., write a `.claim` file with the worker PID and timestamp using `os.open` with `O_CREAT | O_EXCL`). If the file already exists, skip the target. Only write the real sidecar after successful profiling. Clean up claim files on controlled shutdown. Alternatively, pre-partition the target list across workers so each worker owns exclusive targets.

**Detection:** Run a parallel batch and inspect the `batch-summary.json` for duplicate `problem_id` entries or sidecar timestamps that overlap within the same problem.

**Phase:** Phase implementing internal parallelism; also affects any phase touching resume logic.

### Pitfall 3: Stale PID Locks After Crash or OOM Kill

**What goes wrong:** A PID lock file (e.g., `/tmp/sol-execbench-timing-batch.lock` containing a PID) is left behind after the process is killed by OOM, `SIGKILL`, or power loss. A subsequent invocation checks the PID, finds the process no longer exists, but the lock file check is racy or incorrectly implemented, so it either (a) refuses to start (false positive) or (b) deletes the lock and starts (risking concurrent execution with a zombie that still holds GPU resources).

**Why it happens:** PID-based locking is inherently unreliable on Linux. PIDs are recycled. A crashed process's PID can be reused by an unrelated process. Checking `/proc/{pid}` between the crash and the lock deletion creates a window where the lock appears stale but the GPU is still cleaning up the previous process's context.

**Consequences:** False-positive lock blocks all future runs until manual cleanup. Or false-negative allows concurrent GPU execution that corrupts timing (see Pitfall 1). Both are bad for unattended batch runs that may run for hours.

**Prevention:** Use `fcntl.flock` (POSIX advisory lock) or `filelock` library instead of PID-in-file. `flock` is automatically released by the kernel when the file descriptor is closed, including on process death (even `SIGKILL`). Combine with a non-blocking acquire attempt and a clear error message. If PID-based locking must be used (e.g., for NFS or cross-machine coordination), validate the PID against `/proc/{pid}/cmdline` to confirm it is actually the expected script, not a recycled PID.

**Detection:** After a crash, verify the lock is released. Test by `kill -9` on a running instance and confirming the next invocation acquires the lock without manual intervention.

**Phase:** Phase implementing PID locking for multi-instance prevention.

### Pitfall 4: Workload Manifest Corruption Under Concurrent Write

**What goes wrong:** In `run_rdna4_profiler_timing_batch.py`, the workload-sharded path (`_profile_target_workload_sharded`) repeatedly reads, modifies, and writes the workload manifest JSON file. If two threads or processes operate on different workloads of the same problem simultaneously, they can interleave reads and writes, losing one thread's manifest updates.

**Why it happens:** The manifest is a single JSON file per problem. `_load_or_create_workload_manifest` reads it, `_upsert_workload_manifest_entry` modifies the in-memory dict, and `_write_workload_manifest` writes the entire file back. Without per-problem locking, two workers processing different workload slices of the same problem will clobber each other's manifest entries.

**Consequences:** Manifest reports incorrect workload status (e.g., shows workload 3 as incomplete when it was actually completed). Aggregation then reports the problem as incomplete. Sidecar evidence is silently lost from the aggregate.

**Prevention:** If workload-level parallelism is introduced within a problem, add per-problem manifest locking. If the architecture is "one problem at a time on GPU, but CPU staging can overlap," ensure manifest writes happen after the GPU-serial section completes. Alternatively, use per-workload manifest fragments and merge them atomically only after all workloads complete.

**Detection:** Run workload-sharded mode with overlapping CPU staging and inspect manifest entries for missing or overwritten workload indices.

**Phase:** Phase implementing internal parallelism for `run_rdna4_profiler_timing_batch.py`.

### Pitfall 5: ThreadPoolExecutor Exception Swallowing

**What goes wrong:** Exceptions raised inside `concurrent.futures.ThreadPoolExecutor` workers are captured by the `Future` object and only surface when `future.result()` is called. If the orchestration code iterates futures and only logs exceptions without stopping the pool, subsequent work continues on a corrupted or partially initialized state. In the profiler timing batch, this means a staging failure (e.g., invalid definition JSON) silently produces no sidecar, and the problem appears as "not attempted" rather than "failed."

**Why it happens:** ThreadPoolExecutor does not raise exceptions at the point of submission. The `submit()` call returns immediately. If the caller does not promptly call `future.result()` or `future.exception()`, the exception is silently held. This is especially dangerous in batch loops that submit all work and then join.

**Consequences:** Problems silently skipped. Batch summary reports fewer targets than expected. Resume logic may or may not pick them up depending on whether a partial sidecar was written before the exception. Coverage reports show gaps with no explanation.

**Prevention:** Use `as_completed()` or `wait()` with explicit exception checking on every future. Wrap the worker function in a try/except that writes a blocked sidecar (similar to the existing `_write_blocked_sidecar` pattern) even on unexpected exceptions, so every target produces a sidecar. Never let a future go uncollected.

**Detection:** Inject a deliberate exception in one worker and verify the batch summary accounts for it with a proper failure status rather than silently skipping it.

**Phase:** Phase implementing internal parallelism for either script.

## Moderate Pitfalls

### Pitfall 6: Status JSONL Append Race in run_derived_isolated

**What goes wrong:** `append_status()` opens the status JSONL file in append mode and writes a line. Under concurrent access (if `run_derived_isolated.py` is parallelized), two workers can interleave their JSON writes, producing malformed JSONL lines.

**Why it happens:** Append mode (`"a"`) does not guarantee atomic line writes for lines longer than the pipe buffer size (typically 4096 bytes on Linux). `ProblemStatus` JSON serialized with `sort_keys=True` can exceed this for long command arrays.

**Prevention:** Use a threading lock around the status file write. Or use per-worker status files and merge them at the end. Or use `json.dumps()` output that is guaranteed small (the current `ProblemStatus` is usually under 1KB) and rely on POSIX atomic append for writes under PIPE_BUF. The safest approach is a shared lock.

**Phase:** Phase implementing internal parallelism for `run_derived_isolated.py`.

### Pitfall 7: Temp Directory Cleanup Timing

**What goes wrong:** `_staging_runner` creates staging directories via `tempfile.mkdtemp`. In the current serial code, cleanup timing does not matter. With parallel workers, a staging directory created by one worker might be cleaned up by another worker's error handler, or might fill up `/tmp` because many staging directories exist simultaneously.

**Why it happens:** The current code creates a staging directory per target but relies on OS temp directory cleanup or manual cleanup. Parallel execution multiplies the number of simultaneous staging directories.

**Prevention:** Use a per-worker staging directory root (e.g., `temp_dir / worker-{id}`) and clean up the entire worker root when the worker exits. Set a reasonable `temp_dir` default with sufficient space. Add a pre-flight check for available disk space before starting parallel execution.

**Phase:** Phase implementing internal parallelism for `run_rdna4_profiler_timing_batch.py`.

### Pitfall 8: L2 Cache Pollution Between Serial GPU Invocations

**What goes wrong:** Even with GPU-serial execution, if CPU staging work in one thread fills L2/L3 cache while the GPU is executing a profiling subprocess, the next GPU invocation's CPU-side setup (definition loading, workload parsing) may be slower. More importantly, the GPU kernel execution itself is not affected by CPU cache state, but the driver-level command submission and result parsing can have microsecond-level jitter.

**Why it happens:** ThreadPoolExecutor workers share the same process and L3 cache. Heavy JSON parsing in one thread can evict cache lines used by the subprocess result parser in another thread. This is a second-order effect and is unlikely to affect kernel-level timing significantly, but it can affect the orchestration overhead measurements.

**Prevention:** For the profiler timing batch, this is likely negligible because the rocprofv3 tool measures kernel activity directly, not wall-clock time. For wall-clock timing scripts, insert a brief `time.sleep(0)` (thread yield) or explicit cache flush between GPU invocations. Monitor variance before and after introducing parallelism.

**Phase:** Phase implementing internal parallelism; audit during timing reproducibility hardening.

### Pitfall 9: KeyboardInterrupt During Pool Shutdown

**What goes wrong:** If the user sends Ctrl+C while `ThreadPoolExecutor` or `ProcessPoolExecutor` is running, Python raises `KeyboardInterrupt` in the main thread. The default pool shutdown behavior (`wait=True`) deadlocks if workers are blocked on GPU subprocess calls. With `wait=False`, workers continue running in the background and may write partial sidecars.

**Why it happens:** `subprocess.run()` in a worker thread blocks the thread. `KeyboardInterrupt` is only delivered to the main thread in CPython. Worker threads with active subprocess calls continue running. The main thread's pool shutdown waits for them (deadlock) or abandons them (orphaned processes).

**Prevention:** Use `executor.shutdown(wait=False, cancel_futures=True)` (Python 3.9+) in the signal handler. Set a module-level `threading.Event` as a cancellation flag that workers check before starting each new subprocess. Forward `SIGINT` to subprocess groups so rocprofv3/eval_driver processes are terminated promptly. Write partial sidecars with a distinct "interrupted" status so resume logic can distinguish them from completed results.

**Phase:** Phase implementing internal parallelism for either script.

### Pitfall 10: ProcessPoolExecutor Fork Safety with PyTorch/ROCm Imports

**What goes wrong:** Using `ProcessPoolExecutor` with the default `"fork"` start method after PyTorch or ROCm libraries have been imported in the parent process can cause deadlocks, corrupted GPU context, or segfaults in child processes. PyTorch acquires internal locks during initialization that are not released in forked children.

**Why it happens:** Python's `fork()` copies the parent's memory and thread state but not the threads themselves. Any mutex held by a non-forked thread at fork time is permanently locked in the child. PyTorch, HIP runtime, and rocprofv3's underlying libraries all use internal threads that hold locks.

**Prevention:** Use `ThreadPoolExecutor` instead of `ProcessPoolExecutor` for the CPU-parallel staging work. If `ProcessPoolExecutor` is needed (e.g., for true CPU parallelism in derived sidecar generation), use the `"spawn"` or `"forkserver"` start method and ensure no GPU/PyTorch/HIP imports happen in the parent before pool creation. The existing scripts already import from `sol_execbench.core` at module level, which transitively imports PyTorch, so switching to `"spawn"` would require restructuring imports or using lazy imports in worker functions.

**Phase:** Phase implementing internal parallelism; critical decision point for `run_derived_isolated.py`.

### Pitfall 11: Resume Logic Assumes Serial Completion Order

**What goes wrong:** `select_fallback_targets` iterates coverage problems in sorted order and skips those with existing classified sidecars. The serial execution model guarantees that if problem N has a sidecar, all problems before N were also processed. With parallel execution, problems complete out of order. If the process crashes after completing problems 5, 8, and 12 but before completing problem 3, the resume target list on the next run will correctly skip 5, 8, and 12 but will re-attempt problem 3 (correct). However, any state that was built assuming sequential completion (e.g., the batch summary's "selected_targets" count) will be wrong.

**Why it happens:** The `limit` parameter in `select_fallback_targets` stops after `limit` targets. In serial mode, this naturally batches `[0..limit)`. In parallel mode with out-of-order completion, the "first N incomplete" targets on resume may differ from the "first N incomplete" targets on the initial run, because completed targets are interspersed.

**Prevention:** Make resume logic strictly idempotent: always scan all targets and skip completed ones regardless of position. Do not rely on `limit` for correctness (it is a performance optimization only). After parallel execution completes, verify that the final coverage state is identical to what a serial execution would have produced.

**Phase:** Phase implementing internal parallelism; touches resume logic in both scripts.

## Minor Pitfalls

### Pitfall 12: Log File Interleaving

**What goes wrong:** `run_derived_isolated.py` writes all problem logs to a single `--log-file`. With parallel workers, log lines from different problems interleave, making logs harder to read.

**Prevention:** Use per-worker or per-problem log files. Or prefix each line with the problem ID and worker ID for post-hoc sorting.

### Pitfall 13: GIL Not Released by Blocking C Extension

**What goes wrong:** If any imported C extension (PyTorch, HIP runtime) holds the GIL during a blocking operation, `ThreadPoolExecutor` workers will be serialized by the GIL even though they are doing independent work. This defeats the purpose of CPU-parallel staging.

**Prevention:** Verify that the CPU-parallel staging work (JSON parsing, packager construction) does not call into C extensions. The current staging code in `_profile_target` uses pure Python (`json.loads`, Pydantic model construction, `Path` operations), so this should be safe. But verify that `ProblemPackager.__init__` does not trigger GPU initialization.

### Pitfall 14: Over-Parallelizing the Wrong Bottleneck

**What goes wrong:** Profiling reveals that GPU subprocess execution is 95% of wall time and CPU staging is 5%. Parallelizing the staging work saves negligible wall time while adding significant complexity.

**Prevention:** Profile before and after parallelism. If CPU staging is under 10% of total wall time, internal parallelism may not be worth the complexity. Consider whether the real bottleneck is better addressed by other means (e.g., reducing subprocess startup overhead, batching profiler invocations).

### Pitfall 15: Environment Variable Leakage Between Workers

**What goes wrong:** Workers share the same process environment. If one worker modifies `os.environ` (e.g., `environment_with_uv_cache` in `run_derived_isolated.py`), the change is visible to all workers. This can cause unexpected behavior if different workers need different environment configurations.

**Prevention:** Pass environment dictionaries explicitly to `subprocess.run(env=...)` rather than modifying `os.environ`. The existing `_staging_runner` already does this correctly. Ensure new parallel code follows the same pattern.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Internal parallelism for profiler timing batch | GPU subprocess concurrency (Pitfall 1) | CPU-parallel staging, GPU-serial profiling queue |
| Internal parallelism for profiler timing batch | Manifest corruption (Pitfall 4) | Per-problem manifest lock or serial manifest writes |
| Internal parallelism for profiler timing batch | Resume dedup race (Pitfall 2) | Atomic claim files or pre-partitioned target lists |
| Internal parallelism for derived isolated | ProcessPoolExecutor fork safety (Pitfall 10) | Use ThreadPoolExecutor or spawn start method |
| Internal parallelism for derived isolated | Status JSONL append race (Pitfall 6) | Per-worker status lock |
| PID locking implementation | Stale locks after crash (Pitfall 3) | Use flock instead of PID-in-file |
| PID locking implementation | Keyboard interrupt deadlock (Pitfall 9) | Signal handler with cancel_futures and subprocess group kill |
| Timing reproducibility audit | L2 cache pollution (Pitfall 8) | Monitor variance before/after; likely negligible for rocprofv3 kernel activity timing |
| Timing reproducibility audit | Over-parallelizing wrong bottleneck (Pitfall 14) | Profile first; parallelism may not be the right lever |

## Integration Pitfalls

### Existing Contract Preservation

Both `run_batch()` (in `run_rdna4_profiler_timing_batch.py`) and `main()` (in `run_derived_isolated.py`) return process-style exit codes and write structured JSON artifacts. Adding internal parallelism must not change:

1. **Exit code semantics** -- `0` for all-success, `1` for any failure, `2` for argument/IO errors.
2. **Sidecar schema** -- The `.timing.json` sidecar schema must remain identical regardless of whether it was produced serially or with internal parallelism.
3. **Batch summary structure** -- `batch-summary.json` must contain the same fields with the same semantics.
4. **Resume behavior** -- `--resume` must produce identical coverage given the same input state, regardless of execution order.
5. **Deterministic ordering** -- Problem IDs in reports must be sorted deterministically (current code uses `sorted()` on problem paths).

### Subprocess Isolation Boundary

The existing architecture runs GPU work in subprocesses for good reason (isolation, memory cleanup, timeout control). Internal parallelism must preserve this boundary. Do not be tempted to inline the eval driver or profiling logic into the main process for "efficiency." The subprocess boundary is what makes timeout handling, OOM recovery, and GPU context cleanup reliable.

## Sources

- Code analysis of `scripts/run_rdna4_profiler_timing_batch.py` (1417 lines)
- Code analysis of `scripts/run_derived_isolated.py` (324 lines)
- Code analysis of `scripts/run_rdna4_profiler_partial_failures.py` (583 lines)
- Python `concurrent.futures` documentation: ThreadPoolExecutor exception behavior, `shutdown()` semantics
- POSIX `flock(2)` semantics: automatic release on file descriptor close including process termination
- CPython GIL behavior with subprocess and threading
- PyTorch fork safety documentation: internal thread state not preserved across fork
- Linux `O_CREAT | O_EXCL` atomic file creation for lock/claim files
