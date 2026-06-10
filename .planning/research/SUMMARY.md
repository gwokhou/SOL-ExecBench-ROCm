# Project Research Summary

**Project:** SOL ExecBench ROCm -- v1.35 Script Parallelism and Safety Hardening
**Domain:** GPU benchmark orchestration concurrency, multi-instance prevention, timing reproducibility
**Researched:** 2026-06-10
**Confidence:** HIGH

## Executive Summary

This milestone adds internal CPU-parallel / GPU-serial concurrency to two batch benchmark scripts (`run_rdna4_profiler_timing_batch.py` and `run_derived_isolated.py`), PID-based multi-instance prevention, and timing reproducibility hardening. The entire milestone exists because users currently run multiple terminal instances manually to improve throughput, which silently corrupts GPU timing measurements. The fix is to bring parallelism inside the scripts where it can be controlled properly.

Research across all four areas converges on a single architecture: `ThreadPoolExecutor` for CPU-bound staging and subprocess dispatch, a `threading.Semaphore(1)` or serial consumption loop for GPU exclusivity, and `fcntl.flock` for process-level instance locking. Every recommendation uses Python stdlib or already-available PyTorch ROCm APIs -- zero new dependencies. The existing `run_dataset.py` pipeline mode (lines 2417-2453) provides a proven reference implementation that the new code should mirror directly.

The dominant risk is GPU concurrency corrupting profiler timing evidence. The prevention strategy is architecturally enforced: CPU staging runs in parallel threads, but the actual `collect_rocprofv3_timing()` subprocess call is always serialized. Secondary risks include resume-deduplication TOCTOU races under parallel access, manifest file corruption from concurrent writes, and `KeyboardInterrupt` deadlocking the thread pool. All are addressed with specific mitigations documented in PITFALLS.md.

## Key Findings

### Recommended Stack

All capabilities come from Python 3.12+ stdlib and PyTorch ROCm 2.10.0+. No new package installation required.

**Core technologies:**
- `concurrent.futures.ThreadPoolExecutor`: CPU-parallel staging and subprocess dispatch -- already used in `run_dataset.py`, GIL is not a concern because staging work is pure Python I/O and GPU work runs in subprocesses.
- `threading.Semaphore(1)`: GPU-serial execution gate -- simplest composable mechanism to enforce exclusive GPU access while allowing overlapping CPU staging.
- `fcntl.flock(LOCK_EX | LOCK_NB)`: Multi-instance prevention -- kernel-level POSIX advisory lock, auto-released on process death (even SIGKILL), no stale-lock issues, no external dependencies.
- `os.open(O_CREAT | O_EXCL)`: Atomic claim-file creation for resume deduplication -- prevents TOCTOU races without requiring coordination.
- `torch.cuda.empty_cache()` / `torch.cuda.synchronize()`: Inter-problem GPU state cleanup -- already available through PyTorch ROCm, reduces inter-problem timing variance.

### Expected Features

**Must have (table stakes):**
- CPU-parallel staging + GPU-serial profiling for `run_rdna4_profiler_timing_batch.py` -- eliminates the manual multi-instance workflow and its timing bias.
- CPU-parallel dispatch for `run_derived_isolated.py` -- ThreadPoolExecutor around per-problem subprocess dispatch, each problem isolated in its own subprocess.
- `fcntl.flock`-based multi-instance prevention -- prevents concurrent instances that corrupt timing evidence; required for profiler timing batch.
- Atomic resume deduplication via pre-partitioned target lists -- each worker owns exclusive targets by index, no file-based coordination needed.
- Graceful interrupt handling -- cancellation event, `shutdown(wait=False, cancel_futures=True)`, structured `interrupted` sidecar for partial work.
- Deterministic sorted output regardless of parallel completion order.

**Should have (competitive):**
- GPU warmup/idle barrier between serial profiling invocations -- `torch.cuda.synchronize()` + `empty_cache()` between subprocess calls.
- L2 cache pollution audit -- one-time comparison of kernel activity durations with/without parallel staging.
- Per-worker staging directory isolation -- `temp_dir / worker-{id}` root per worker.
- Structured cancellation sidecar -- `replacement_status: "interrupted"` distinguishable from `profiler_blocked`.

**Defer (v2+):**
- Configurable `--workers` CLI flag -- hard-code reasonable defaults internally for v1.
- Multi-GPU parallel profiling -- requires fundamental architecture changes, separate milestone.
- Rich progress bar integration -- post-processing on batch summary, not during execution.

### Architecture Approach

The architecture introduces two new small modules (`pid_lock.py`, `timing_isolation.py`) in `core/bench/` and modifies three existing scripts. The CPU-parallel-prepare / serial-GPU-execute pattern from `run_dataset.py` pipeline mode is replicated directly. The subprocess isolation boundary is preserved -- GPU work never moves into the main process.

**Major components:**
1. `pid_lock.py` (NEW) -- `fcntl.flock`-based context manager for multi-instance prevention; zero dependencies, kernel-managed lifecycle.
2. `timing_isolation.py` (NEW) -- Pre-flight GPU contention detection, clock lock verification, environment audit; advisory-only by default.
3. `run_rdna4_profiler_timing_batch.py` (MODIFY) -- Split `_profile_target` into prepare (CPU) and execute (GPU) phases, add `ThreadPoolExecutor` for prepare, add PID lock and timing isolation audit at entry.
4. `run_derived_isolated.py` (MODIFY) -- Wrap serial for-loop in `ThreadPoolExecutor.map`, add `threading.Lock` around status file writes.
5. `evaluation_stability.py` (MODIFY) -- Add reason codes for `gpu_contention`, `multi_instance_interference`, `l2_cache_pollution_risk`.

### Critical Pitfalls

1. **GPU subprocess concurrency destroys timing validity** -- Never run two `rocprofv3` calls concurrently. Enforce CPU-parallel staging + GPU-serial execution architecturally. Detection: compare kernel activity durations with/without parallelism.
2. **File-based resume deduplication TOCTOU race** -- Two workers both see a target as incomplete and both process it. Prevention: pre-partition target lists by worker index, or use atomic `O_CREAT | O_EXCL` claim files.
3. **Stale PID locks after crash** -- PID-based locking is unreliable (PIDs recycle). Prevention: use `fcntl.flock` which is auto-released by the kernel on process death. Never implement stale-PID detection.
4. **Workload manifest corruption under concurrent write** -- Per-problem manifest JSON is a single file. Prevention: ensure manifest writes happen after the GPU-serial section completes, or add per-problem manifest locking.
5. **ThreadPoolExecutor exception swallowing** -- Exceptions in workers are held by `Future` objects and silently dropped if `future.result()` is never called. Prevention: wrap workers in try/except that always produces a sidecar (blocked or interrupted).

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: PID Lock Module
**Rationale:** Zero dependencies, testable in isolation, required by Phase 3. Build the foundation first.
**Delivers:** `pid_lock.py` in `core/bench/` with unit tests. Context manager with non-blocking acquire, diagnostic error messages, auto-release on process death.
**Addresses:** Multi-instance prevention (FEATURES.md table stakes), stale lock avoidance (PITFALLS.md Pitfall 3).
**Avoids:** No pitfall exposure -- pure stdlib, no GPU, no parallelism.
**Research flag:** Standard pattern -- well-documented `fcntl.flock` usage, no additional research needed.

### Phase 2: Timing Isolation Audit + Derived Script Parallelism
**Rationale:** Timing isolation is a dependency for Phase 3. Derived script parallelism is the lowest-risk concurrency change (CPU-only, no GPU concerns) and provides immediate throughput value.
**Delivers:** `timing_isolation.py` module. `run_derived_isolated.py` with `ThreadPoolExecutor` dispatch, `--jobs` flag, thread-safe status writes.
**Uses:** `concurrent.futures.ThreadPoolExecutor`, `threading.Lock` for status file.
**Implements:** CPU-parallel dispatch for derived sidecars (ARCHITECTURE.md Pattern 2).
**Avoids:** ProcessPoolExecutor fork safety issue (PITFALLS.md Pitfall 10), status JSONL append race (Pitfall 6), KeyboardInterrupt deadlock (Pitfall 9).
**Research flag:** Standard pattern -- mirrors existing `run_dataset.py` derived parallel mode (lines 2487-2523).

### Phase 3: Profiler Timing Batch Parallelism
**Rationale:** The most complex change. Requires refactoring `_profile_target` into prepare/execute phases, integrating PID lock from Phase 1, timing isolation from Phase 2, and ThreadPoolExecutor for staging. Must be built on top of proven infrastructure.
**Delivers:** `run_rdna4_profiler_timing_batch.py` with CPU-parallel staging, GPU-serial profiling, PID lock acquisition, atomic resume deduplication, graceful interrupt handling, deterministic output.
**Uses:** All stack elements: `ThreadPoolExecutor`, `Semaphore(1)`, `pid_lock`, `timing_isolation`, atomic claim files.
**Implements:** CPU-parallel prepare / serial GPU execute (ARCHITECTURE.md Pattern 1).
**Avoids:** GPU concurrency (Pitfall 1), resume dedup race (Pitfall 2), manifest corruption (Pitfall 4), exception swallowing (Pitfall 5), resume order assumption (Pitfall 11).
**Research flag:** Needs careful planning -- the split of `_profile_target` into prepare/execute requires understanding the full 1417-line script. Recommend `/gsd:plan-phase --research-phase 3`.

### Phase 4: Evaluation Stability Extension + Integration Testing
**Rationale:** After all parallelism changes are in place, extend stability diagnostics to capture new failure modes and validate the complete system end-to-end.
**Delivers:** New reason codes in `evaluation_stability.py`. Integration tests for PID lock contention, parallel prepare + serial execute, isolation audit output.
**Addresses:** Diagnostic coverage for new concurrency failure modes.
**Research flag:** Standard pattern -- extending existing diagnostic module.

### Phase 5: GPU Warmup Barrier + L2 Audit (Post-Validation)
**Rationale:** Quality-of-life improvements that depend on the parallel infrastructure being proven in production. These are P2 features from the prioritization matrix.
**Delivers:** Inter-problem GPU state cleanup, one-time L2 cache pollution audit comparing kernel activity durations with/without parallel staging.
**Research flag:** May need empirical validation -- L2 cache behavior on RDNA4 is hardware-specific.

### Phase Ordering Rationale

- **Dependencies first:** `pid_lock.py` has zero deps and is imported by Phase 3. `timing_isolation.py` depends on existing `clock_lock.py` and is called at Phase 3 startup.
- **Lowest risk first:** Phase 2 (derived script, CPU-only) validates the ThreadPoolExecutor + interrupt handling patterns before the complex GPU-serial Phase 3.
- **Complex change isolated:** Phase 3 is the only phase that touches GPU execution flow. By building it last, all supporting infrastructure is already tested.
- **How this avoids pitfalls:** GPU concurrency is architecturally impossible (semaphore + PID lock). TOCTOU races are prevented by pre-partitioned targets. Exception swallowing is handled by the try/except pattern established in Phase 2.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Complex refactoring of 1417-line script; needs detailed analysis of `_profile_target` function boundaries, manifest write timing, and resume state management under parallelism.
- **Phase 5:** L2 cache behavior on RDNA4 is hardware-specific; may need empirical measurement to validate the audit methodology.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Well-documented `fcntl.flock` usage, direct code sketch provided in ARCHITECTURE.md.
- **Phase 2:** Direct replication of existing `run_dataset.py` derived parallel mode.
- **Phase 4:** Extending existing diagnostic module with new string constants and wiring.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technologies are Python stdlib or already-available PyTorch APIs. Zero new dependencies. Reference implementation exists in `run_dataset.py`. |
| Features | HIGH | Feature list derived from direct code analysis of target scripts. Priorities validated against existing codebase patterns. Anti-features identified with clear rationale. |
| Architecture | HIGH | All conclusions drawn from direct codebase inspection. Component boundaries, data flow, and integration points verified against source. Reference pattern (pipeline mode) already in production. |
| Pitfalls | HIGH | Pitfalls derived from code analysis of existing scripts and well-established Python concurrency engineering knowledge. Prevention strategies are concrete and implementable. |

**Overall confidence:** HIGH

### Gaps to Address

- **Actual throughput benefit of CPU-parallel staging:** Research recommends profiling before/after to validate that CPU staging is a meaningful fraction of wall time (PITFALLS.md Pitfall 14). If staging is under 10% of total time, parallelism adds complexity without benefit. Address during Phase 3 planning with empirical measurement.
- **RDNA4 L2 cache behavior under CPU-parallel staging:** The L2 cache pollution audit (Phase 5) requires understanding whether CPU-side JSON parsing in parallel threads measurably affects GPU kernel timing. This is hardware-specific and needs empirical validation.
- **Thread safety of `ProblemPackager.__init__`:** Research assumes pure Python construction is GIL-safe, but should verify that `ProblemPackager` does not trigger GPU initialization or C extension calls during staging. Address during Phase 3 implementation with a code audit.

## Sources

### Primary (HIGH confidence)
- Code analysis of `scripts/run_dataset.py` -- existing ThreadPoolExecutor patterns (pipeline mode lines 2417-2453, derived parallel lines 2487-2523)
- Code analysis of `scripts/run_rdna4_profiler_timing_batch.py` (1417 lines) -- target scripts, staging, profiling, manifest, resume patterns
- Code analysis of `scripts/run_derived_isolated.py` (324 lines) -- subprocess dispatch, resume, launch-mode patterns
- Code analysis of `src/sol_execbench/core/bench/timing.py` -- GPU event timing, L2 cache flush, warmup patterns
- Code analysis of `src/sol_execbench/core/bench/clock_lock.py` -- STABLE_PEAK clock locking
- Code analysis of `src/sol_execbench/core/bench/rocm_profiler.py` -- injectable ProfilerRunner
- Python stdlib `concurrent.futures` documentation (Python 3.12+)
- POSIX `fcntl(2)` / `flock(2)` semantics

### Secondary (MEDIUM confidence)
- PyTorch fork safety documentation -- internal thread state not preserved across fork
- CPython GIL behavior with subprocess and threading
- PyPI `filelock` 3.29.0 -- researched and rejected in favor of stdlib

---
*Research completed: 2026-06-10*
*Ready for roadmap: yes*
