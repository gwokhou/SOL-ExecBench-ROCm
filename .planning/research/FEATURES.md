# Feature Research

**Domain:** Internal CPU-parallel/GPU-serial concurrency, PID-based instance locking, and execution environment independence for statistics-sensitive timing scripts in a ROCm GPU benchmarking project.
**Researched:** 2026-06-10
**Confidence:** HIGH (based on code analysis of existing scripts and well-established Python concurrency, file-locking, and GPU benchmarking patterns)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist for safe, reproducible batch benchmark execution. Missing any of these = the batch timing evidence is unreliable or the user accidentally corrupts a multi-hour run.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| CPU-parallel staging with GPU-serial profiling in `run_rdna4_profiler_timing_batch.py` | Users currently must run multiple terminal instances manually to get throughput, which introduces multi-instance timing bias. Internal parallelism should eliminate that manual workflow and its bias. | MEDIUM | Stage definition loading, workload slicing, and `ProblemPackager` construction in a `ThreadPoolExecutor`. Serialize the actual `collect_rocprofv3_timing()` call through a single GPU worker. The CPU staging work is pure Python (JSON parsing, Pydantic model construction, Path operations) -- no GIL contention. The existing `_profile_target` function already separates staging from the subprocess call. |
| CPU-parallel problem dispatch in `run_derived_isolated.py` | Users run 100+ derived sidecar generations sequentially, each taking seconds to minutes. Multi-problem parallelism is the obvious throughput win because each problem runs in an isolated subprocess that does its own GPU work internally. | LOW | The script already dispatches per-problem `subprocess.run()` calls. Add `ThreadPoolExecutor` around the dispatch loop. Each problem's subprocess is already isolated (own `uv run`, own staging, own memory). GPU serialization is enforced by the subprocess-level `sol-execbench` CLI which already handles one problem at a time per invocation. Parallelism here is CPU-side subprocess orchestration only. |
| Defensive multi-instance prevention via file lock | The entire v1.35 milestone exists because users ran multiple instances and got timing bias. Any script that touches GPU timing MUST prevent accidental concurrent execution. | LOW | Use `fcntl.flock` (POSIX advisory lock) on a well-known lock file. `flock` is automatically released by the kernel on process death (including `SIGKILL`), avoiding stale lock issues. This is a Python stdlib solution -- no new dependencies. Wrap in a context manager for clean acquire/release. Required for `run_rdna4_profiler_timing_batch.py` (GPU profiling). Optional but recommended for `run_derived_isolated.py` (subprocess GPU work). Not needed for CPU-only report/analysis scripts like `run_rdna4_profiler_partial_failures.py`. |
| Atomic resume deduplication | The existing `--resume` flag uses non-atomic check-then-write (`_is_classified_replacement_sidecar` then later write). Under parallel execution, two workers can both see a target as incomplete and both process it. | LOW | Pre-partition the target list across workers (simplest, no coordination needed). Each worker owns exclusive targets by index. The existing `select_fallback_targets` already supports `limit` and `skip` filtering -- extend to support partition-by-index. No file-based locking needed for target deduplication. |
| Graceful interrupt handling with pool cleanup | Long batch runs (hours) will be interrupted. Ungraceful shutdown leaves partial sidecars that confuse resume logic. | MEDIUM | Register `SIGINT`/`SIGTERM` handler that sets a `threading.Event` cancellation flag. Workers check the flag before each new target. Use `executor.shutdown(wait=False, cancel_futures=True)` (Python 3.9+). Write partial sidecars with a distinct `interrupted` status so resume logic can distinguish them from completed results. The existing `_write_blocked_sidecar` pattern can be extended with an `interrupted` reason. |
| Deterministic output regardless of execution order | Users and CI compare batch summaries across runs. Output must be identical whether problems completed in order [1,2,3] or [3,1,2]. | LOW | The existing code already sorts results by `problem_id` in most output paths. Extend this guarantee to all parallel output: sort `results` list before writing `batch-summary.json`, sort status JSONL entries by problem_id on merge. The sidecar schema itself is per-problem and already deterministic. |

### Differentiators (Competitive Advantage)

Features that set the benchmark infrastructure apart. Not required, but valuable for timing evidence quality and operational trustworthiness.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| GPU warmup/idle barrier between serial profiling invocations | Ensures each profiling subprocess starts with a clean GPU state (no residual L2 cache, no pending async operations from the previous subprocess). Reduces inter-problem timing variance. | LOW | Insert `torch.cuda.synchronize()` + optional `torch.cuda.empty_cache()` between GPU subprocess invocations. The existing `bench_time_with_device_events` already does `_clear_cache(cache)` before each iteration -- extend this pattern to the inter-problem boundary in the batch script. Only relevant for `run_rdna4_profiler_timing_batch.py` where the same GPU processes multiple problems serially. |
| L2 cache pollution audit | Quantifies whether CPU-parallel staging work affects GPU timing variance by comparing kernel activity durations with and without parallel staging enabled. | LOW | Run the same problem set twice: once with internal parallelism, once without. Compare kernel activity duration distributions. Flag if coefficient of variation increases beyond the existing `evaluation_stability` threshold. This is a one-time audit, not a permanent runtime check. Record results in the evaluation stability sidecar. |
| Per-worker staging directory isolation | Prevents temp directory conflicts and enables clean per-worker cleanup. | LOW | Use `temp_dir / f"worker-{worker_id}"` as the staging root for each worker. Clean up the worker root when the worker exits (success or failure). The existing `_staging_runner` already creates per-target staging directories via `tempfile.mkdtemp` -- just change the `dir=` argument to a per-worker root. |
| Structured cancellation sidecar | When a run is interrupted, write a sidecar with `replacement_status: "interrupted"` and partial results, so resume logic can distinguish interrupted work from never-attempted or failed work. | LOW | Extend `_write_blocked_sidecar` with an `interrupted` status. The resume check in `_is_classified_replacement_sidecar` should skip interrupted sidecars (they are retryable). This is more granular than the existing `profiler_blocked` status. |
| Environment capture in lock context | When a PID lock is acquired, record which script, which GPU, and which user holds the lock, so a blocked user can diagnose who is using the GPU. | LOW | Write a small JSON sidecar next to the lock file containing `{"pid": ..., "command": ..., "gpu_arch": ..., "acquired_at": ...}`. Clean up on release. This is diagnostic only -- the actual lock mechanism is `flock`, not the file contents. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems for this specific domain.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Multi-GPU parallel profiling | "Why not profile multiple problems on multiple GPUs simultaneously?" | ROCm GPU context is per-device. The existing scripts assume a single GPU device. Multi-GPU support requires device selection, per-device rocprofv3 invocation, and per-device clock lock management. The current codebase has no multi-GPU awareness. Adding it would require changes to `ProblemPackager`, `collect_rocprofv3_timing`, `_staging_runner`, and every caller. This is a different milestone. | Single-GPU serial profiling with CPU-parallel staging. If multi-GPU is needed later, add a device parameter to the batch script and run one instance per GPU with separate lock files. |
| ProcessPoolExecutor for CPU staging | "ProcessPoolExecutor gives true CPU parallelism unlike ThreadPoolExecutor under the GIL." | The existing scripts import from `sol_execbench.core` at module level, which transitively imports `torch` (PyTorch ROCm). `ProcessPoolExecutor` with the default `fork` start method after PyTorch import causes deadlocks and corrupted GPU contexts (PyTorch internal locks are not released in forked children). Using `spawn` requires restructuring all imports. | ThreadPoolExecutor is correct here because the CPU staging work (JSON parsing, Pydantic model construction, Path operations) is pure Python and does not hold the GIL. The GIL is only relevant for CPU-bound C extensions, which the staging code does not use. The actual heavy work (GPU profiling) runs in subprocesses, which are inherently GIL-free. |
| PID-in-file locking | "Write a file with the PID number and check if the process exists." | PIDs are recycled on Linux. A crashed process's PID can be reused by an unrelated process, causing false negatives (lock appears stale but GPU is still cleaning up) or false positives (unrelated process holds the PID, blocking the script). Stale PID files require manual cleanup after crashes. | Use `fcntl.flock` (POSIX advisory lock). The kernel automatically releases the lock when the file descriptor is closed, including on process death (`SIGKILL`, OOM, power loss). No stale lock files. No PID recycling issues. Python stdlib, no new dependencies. |
| GPU memory pre-allocation barrier | "Pre-allocate all GPU memory before profiling to prevent allocation jitter." | The existing eval driver and rocprofv3 tooling manage their own GPU memory. Pre-allocating all memory would cause OOM failures in the profiling subprocess. The benchmark's `ShiftingMemoryPoolAllocator` already handles memory pre-allocation within each problem's execution. | The existing `_clear_cache` pattern (L2 cache flush via dummy tensor) is sufficient. Do not pre-allocate across problem boundaries. |
| Configurable parallelism degree via CLI flag | "Let users choose how many parallel staging workers to use." | Over-engineering for the current use case. The CPU staging work is lightweight (JSON parsing, model construction). Even 2 workers provide near-maximal throughput improvement. Exposing a `--workers` flag invites users to set it too high, consuming memory and file descriptors without benefit. The GPU side remains serial regardless. | Default to `min(len(targets), os.cpu_count() or 4, 8)` internally. Hard-code a reasonable max (8 workers). Do not expose a CLI flag in v1. If profiling shows a strong need, add it in a later version. |
| Real-time progress reporting with progress bars | "Show a Rich progress bar during batch execution." | The scripts write to stdout and log files. Adding Rich progress bars to a parallel executor requires careful integration with the executor's completion callbacks and can interfere with JSONL output parsing. The current `print(f"DERIVED {problem_id}", flush=True)` pattern is sufficient for monitoring. | Keep the existing print-per-problem pattern. If rich reporting is desired, add it as a post-processing step on the batch summary, not during execution. |

## Feature Dependencies

```
[CPU-parallel staging (profiler timing batch)]
    requires --> [GPU-serial profiling queue]
    requires --> [Atomic resume deduplication]
    requires --> [Graceful interrupt handling]
    enhances --> [Per-worker staging directory isolation]

[CPU-parallel dispatch (derived isolated)]
    requires --> [Graceful interrupt handling]
    requires --> [Atomic resume deduplication]
    enhances --> [Per-worker staging directory isolation]

[PID-based file lock (flock)]
    standalone -- no dependencies on parallelism features
    required-by --> [CPU-parallel staging (profiler timing batch)]

[GPU warmup/idle barrier]
    requires --> [GPU-serial profiling queue]
    enhances --> [L2 cache pollution audit]

[L2 cache pollution audit]
    requires --> [CPU-parallel staging (profiler timing batch)]
    requires --> [GPU warmup/idle barrier]

[Structured cancellation sidecar]
    requires --> [Graceful interrupt handling]
    enhances --> [Atomic resume deduplication]

[Environment capture in lock context]
    requires --> [PID-based file lock (flock)]

[ProcessPoolExecutor for CPU staging] --conflicts--> [Existing torch/ROCm imports at module level]
[PID-in-file locking] --conflicts--> [PID recycling on Linux]
```

### Dependency Notes

- **CPU-parallel staging requires GPU-serial profiling queue:** The entire reason for the GPU-serial constraint is that concurrent rocprofv3 or eval_driver invocations on the same GPU produce meaningless timing. The profiling subprocess call MUST be serialized even while staging is parallel.
- **PID-based file lock required by profiler timing batch:** This script produces timing evidence that must be reproducible. Concurrent instances corrupt that evidence. The lock prevents accidental concurrent execution by the user or by automated tooling.
- **Graceful interrupt handling required by both parallel scripts:** Without it, Ctrl+C on a multi-hour batch run either deadlocks (pool waits for workers blocked on subprocess) or orphans GPU processes. Both outcomes require manual cleanup.
- **ProcessPoolExecutor conflicts with existing torch imports:** `run_rdna4_profiler_timing_batch.py` imports `from sol_execbench.core import BenchmarkConfig, Definition, Solution, Workload` and `from sol_execbench.core.bench.rocm_profiler import ...` at module level. These transitively import `torch`. Using `ProcessPoolExecutor` with fork after torch import is unsafe. ThreadPoolExecutor is the correct choice.

## MVP Definition

### Launch With (v1)

Minimum viable internal parallelism and safety hardening.

- [ ] CPU-parallel staging + GPU-serial profiling for `run_rdna4_profiler_timing_batch.py` -- eliminates the manual multi-instance workflow and its timing bias. ThreadPoolExecutor for staging, single GPU worker for `collect_rocprofv3_timing()` calls.
- [ ] CPU-parallel dispatch for `run_derived_isolated.py` -- ThreadPoolExecutor around the per-problem `subprocess.run()` dispatch loop. Each problem remains isolated in its own subprocess.
- [ ] `fcntl.flock`-based multi-instance prevention for `run_rdna4_profiler_timing_batch.py` -- prevents accidental concurrent execution that corrupts timing evidence. Context manager wrapper, non-blocking acquire, clear error message.
- [ ] Atomic resume deduplication via pre-partitioned target lists -- extend `select_fallback_targets` to support worker partition by index. Each worker owns exclusive targets. No coordination needed.
- [ ] Graceful interrupt handling -- cancellation event, `shutdown(wait=False, cancel_futures=True)`, structured `interrupted` sidecar for partially-completed work.
- [ ] Deterministic sorted output regardless of parallel completion order.

### Add After Validation (v1.x)

Features to add once core parallelism is working and validated.

- [ ] GPU warmup/idle barrier between serial profiling invocations -- reduces inter-problem timing variance.
- [ ] L2 cache pollution audit -- one-time comparison of kernel activity durations with and without parallel staging.
- [ ] Per-worker staging directory isolation -- cleaner temp directory management.
- [ ] Environment capture in lock context -- diagnostic sidecar for blocked users.

### Future Consideration (v2+)

Features to defer until the parallel infrastructure is proven in production use.

- [ ] Configurable `--workers` CLI flag -- only if profiling shows a strong need beyond the default.
- [ ] Multi-GPU parallel profiling -- requires fundamental architecture changes (per-device context, per-device clock lock, per-device rocprofv3).
- [ ] Rich progress bar integration -- post-processing on batch summary, not during execution.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| CPU-parallel staging + GPU-serial profiling (timing batch) | HIGH | MEDIUM | P1 |
| CPU-parallel dispatch (derived isolated) | HIGH | LOW | P1 |
| `flock` multi-instance prevention (timing batch) | HIGH | LOW | P1 |
| Atomic resume deduplication via pre-partition | HIGH | LOW | P1 |
| Graceful interrupt handling | HIGH | MEDIUM | P1 |
| Deterministic output ordering | MEDIUM | LOW | P1 |
| GPU warmup/idle barrier | MEDIUM | LOW | P2 |
| L2 cache pollution audit | MEDIUM | LOW | P2 |
| Per-worker staging directory isolation | LOW | LOW | P2 |
| Environment capture in lock context | LOW | LOW | P3 |
| Structured cancellation sidecar | MEDIUM | LOW | P2 |
| Configurable `--workers` flag | LOW | LOW | P3 |
| Multi-GPU parallel profiling | MEDIUM | HIGH | P3 (different milestone) |

**Priority key:**
- P1: Must have for launch -- without these, parallel execution is unsafe or unreliable
- P2: Should have -- improves evidence quality and operational convenience
- P3: Nice to have, future consideration

## Existing Infrastructure Leveraged

This section documents what already exists and does NOT need to be built from scratch.

| Existing Feature | Where | How It Supports New Features |
|------------------|-------|------------------------------|
| `subprocess.run()` for GPU work | Both scripts | The subprocess boundary already provides isolation. Parallelism adds ThreadPoolExecutor around the orchestration, not the GPU work itself. |
| `--resume` flag with file-based dedup | Both scripts | The resume mechanism works; it just needs atomic guarantees under parallel access (pre-partitioned targets). |
| `ProblemPackager` for staging | `run_rdna4_profiler_timing_batch.py` | Pure Python construction -- safe to run in parallel threads. No GPU state touched during staging. |
| `_staging_runner()` with explicit `env=` | `run_rdna4_profiler_timing_batch.py` | Already passes environment explicitly to subprocess, not via `os.environ` mutation. Safe for parallel workers. |
| `_write_blocked_sidecar()` for error cases | `run_rdna4_profiler_timing_batch.py` | Pattern for writing structured failure sidecars. Extend with `interrupted` status. |
| `ProfilerRunner` injectable runner | `sol_execbench.core.bench.rocm_profiler` | Already supports dependency injection for testing. GPU-serial queue can wrap the runner. |
| `select_fallback_targets()` with `limit`/`skip` | `run_rdna4_profiler_timing_batch.py` | Already supports filtered target selection. Extend with `partition_index`/`partition_count` parameters. |
| `should_skip()` for resume | `run_derived_isolated.py` | Already supports skip-by-completed-set. Thread-safe when each worker has exclusive targets. |
| `torch.cuda.synchronize()` and `_clear_cache()` | `sol_execbench.core.bench.timing` | Existing GPU synchronization patterns for inter-problem barriers. |
| `clock_lock.py` for `STABLE_PEAK` | `sol_execbench.core.bench` | GPU clock state is already managed. The lock should be held for the entire batch duration. |

## Scripts NOT Requiring Changes

Per the milestone scope, these scripts are excluded because they are purely report/analysis utilities with no GPU interaction:

| Script | Why Excluded |
|--------|-------------|
| `scripts/run_rdna4_profiler_partial_failures.py` | Pure classification/reporting. Reads existing timing sidecars and writes classification JSON/Markdown. No GPU, no subprocess execution, no timing measurement. |
| `scripts/report_evaluation_stability.py` | Reporting script. Analyzes existing evaluation stability sidecars. No GPU interaction. |

## Sources

- Code analysis of `scripts/run_rdna4_profiler_timing_batch.py` (1417 lines): staging, profiling, manifest, and resume patterns
- Code analysis of `scripts/run_derived_isolated.py` (324 lines): subprocess dispatch, resume, and launch-mode patterns
- Code analysis of `scripts/run_rdna4_profiler_partial_failures.py` (583 lines): classification-only, no GPU, excluded from scope
- Code analysis of `src/sol_execbench/core/bench/timing.py`: GPU event timing, L2 cache flush, warmup patterns
- Code analysis of `src/sol_execbench/core/bench/clock_lock.py`: STABLE_PEAK clock locking mechanism
- Code analysis of `src/sol_execbench/core/bench/rocm_profiler.py`: injectable ProfilerRunner, rocprofv3 collection
- Code analysis of `src/sol_execbench/core/evaluation_stability.py`: stability diagnostic patterns
- Python `concurrent.futures` documentation: ThreadPoolExecutor, exception propagation, shutdown semantics
- POSIX `flock(2)`: automatic release on file descriptor close, non-blocking acquire via `LOCK_EX | LOCK_NB`
- PyTorch fork safety: internal thread state not preserved across fork, deadlock risk after torch import

---
*Feature research for: script parallelism and safety hardening (v1.35)*
*Researched: 2026-06-10*
