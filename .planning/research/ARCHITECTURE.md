# Architecture Research: Script Parallelism and Safety Hardening

**Domain:** Internal concurrency, multi-instance prevention, and timing isolation for statistics-sensitive benchmark scripts
**Researched:** 2026-06-10
**Confidence:** HIGH (all conclusions drawn from direct codebase inspection)

## System Overview

```
+---------------------------------------------------------------------+
|                     Script Layer (scripts/)                          |
|  +------------------+  +-------------------+  +------------------+  |
|  | run_dataset.py   |  | run_rdna4_profiler |  | run_derived_     |  |
|  | (reference       |  | _timing_batch.py   |  | isolated.py      |  |
|  |  pattern)        |  | (target - add      |  | (target - add    |  |
|  |                  |  |  CPU parallelism)  |  |  CPU parallelism)|  |
|  +--------+---------+  +--------+----------+  +--------+---------+  |
|           |                      |                       |           |
+-----------|----------------------|-----------------------|-----------+
            |                      |                       |
+-----------|----------------------|-----------------------|-----------+
|           v                      v                       v           |
|  +------------------+  +--------+----------+  +---------+---------+  |
|  | [NEW] pid_lock   |  | [EXISTING]         |  | [EXISTING]        |  |
|  | module           |  | core/bench/         |  | core/dataset/     |  |
|  | (new component)  |  | rocm_profiler.py    |  | runner.py         |  |
|  +------------------+  +-------------------+  +------------------+  |
|                                                                      |
|  +------------------+  +-------------------+  +------------------+  |
|  | [NEW] timing_    |  | [EXISTING]         |  | [EXISTING]        |  |
|  | isolation.py     |  | core/bench/         |  | core/bench/       |  |
|  | (new component)  |  | clock_lock.py       |  | config/           |  |
|  +------------------+  +-------------------+  +------------------+  |
+---------------------------------------------------------------------+
            |
            v
+---------------------------------------------------------------------+
|               GPU Runtime (single GPU, serial access)                |
|  +------------------+  +-------------------+  +------------------+  |
|  | rocprofv3        |  | PyTorch ROCm       |  | amd-smi           |  |
|  | profiler         |  | eval subprocess    |  | clock control     |  |
|  +------------------+  +-------------------+  +------------------+  |
+---------------------------------------------------------------------+
```

## Component Responsibilities

### Existing Components (modified)

| Component | Location | Modification Needed |
|-----------|----------|---------------------|
| `run_rdna4_profiler_timing_batch.py` | `scripts/` | Replace serial target loop with CPU-parallel prepare + serial GPU execution |
| `run_derived_isolated.py` | `scripts/` | Replace serial for-loop with ThreadPoolExecutor for CPU-bound sidecar generation |
| `evaluation_stability.py` | `src/sol_execbench/core/` | Extend reason codes for multi-instance interference, L2 cache pollution, GPU contention |

### New Components (added)

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `pid_lock.py` | `src/sol_execbench/core/bench/` | POSIX fcntl-based PID lock for multi-instance prevention |
| `timing_isolation.py` | `src/sol_execbench/core/bench/` | GPU isolation checks, L2 cache warm cache eviction hints, environment independence audit |

## Recommended Project Structure

```
src/sol_execbench/core/bench/
+-- __init__.py
+-- clock_lock.py           # EXISTING - GPU clock lock/unlock/verify
+-- config/
|   +-- benchmark_config.py # EXISTING - warmup, iterations, lock_clocks
+-- pid_lock.py             # NEW - fcntl advisory lock + PID validation
+-- timing_isolation.py     # NEW - GPU contention detection, L2 audit, env checks
+-- rocm_profiler.py        # EXISTING - rocprofv3 collection
+-- timing_policy.py        # EXISTING - timing backend/source policy
+-- ...
```

### Structure Rationale

- `pid_lock.py` lives in `core/bench/` because it is a benchmark execution guard, not a CLI concern. Scripts import it at startup and acquire before any GPU work begins.
- `timing_isolation.py` lives in `core/bench/` alongside `clock_lock.py` because it serves the same role: pre-flight environment validation for timing-sensitive execution.
- Both are importable by scripts AND by the eval driver subprocess if future per-problem isolation checks are needed.

## Architectural Patterns

### Pattern 1: CPU-Parallel Prepare, Serial GPU Execute

**What:** Use `ThreadPoolExecutor` for CPU-bound staging (JSON parsing, packager construction, file I/O) while keeping GPU subprocess execution strictly serial. This is the pattern already proven in `run_dataset.py` via `--execution-mode pipeline`.

**When to use:** For `run_rdna4_profiler_timing_batch.py` which currently iterates targets in a serial for-loop at line 125, doing CPU-bound `ProblemPackager` construction before each serial `rocprofv3` subprocess call.

**Trade-offs:**
- (+) Overlaps staging I/O with GPU execution without any GPU contention
- (+) Follows the established `run_dataset.py` pattern exactly
- (-) Adds complexity to result ordering and progress logging
- (-) Requires careful error propagation from prepare futures

**Existing reference:**
```python
# From run_dataset.py lines 2417-2453 (pipeline mode)
with ThreadPoolExecutor(max_workers=prepare_jobs) as executor:
    future_to_index = {
        executor.submit(
            _prepare_pipeline_trace_problem, ...
        ): index
        for index, problem_dir in enumerate(problems)
    }
    for future in as_completed(future_to_index):
        prepared = future.result()
        result = _run_pipeline_trace_problem(prepared=prepared, ...)
        # serial GPU execution happens here
```

**Proposed for `run_rdna4_profiler_timing_batch.py`:**
```python
# New pattern: prepare targets in parallel, execute GPU serially
with ThreadPoolExecutor(max_workers=prepare_jobs) as executor:
    prepared_futures = {
        executor.submit(
            _prepare_profiler_target, target, dataset_root=dataset_root, ...
        ): target.problem_id
        for target in targets
    }
    for future in as_completed(prepared_futures):
        prepared = future.result()
        results.append(
            _profile_target(prepared, ...)  # serial GPU subprocess
        )
```

### Pattern 2: ThreadPoolExecutor for CPU-Bound Derived Sidecars

**What:** Replace the serial for-loop in `run_derived_isolated.py` (lines 291-319) with `ThreadPoolExecutor.map` for parallel per-problem subprocess execution. Since `--phase derived` only runs CPU-bound sidecar generation (AMD score, SOLAR derivation, SOL bounds) from existing traces, there is no GPU contention risk.

**When to use:** `run_derived_isolated.py` with `--phase derived`, which is already CPU-only work delegated to `run_dataset.py --phase derived` subprocesses.

**Trade-offs:**
- (+) Directly follows the proven `run_dataset.py --phase derived` parallel pattern (lines 2487-2523)
- (+) No GPU contention: derived phase only reads traces and writes JSON sidecars
- (-) Subprocess spawning overhead is the bottleneck, not CPU; gains are modest for small batches
- (-) Shared log file requires synchronization

**Existing reference:**
```python
# From run_dataset.py lines 2514-2523 (derived parallel mode)
with ThreadPoolExecutor(max_workers=jobs) as executor:
    results = executor.map(run_derived_item, enumerate(problems))
    for result in results:
        for message in result["messages"]:
            print(message)
```

**Proposed for `run_derived_isolated.py`:**
```python
# Replace the serial for-loop at line 291
with ThreadPoolExecutor(max_workers=jobs) as executor:
    def run_one(problem_dir):
        problem_id = problem_id_for(benchmark_dir, problem_dir)
        if should_skip(problem_id, ...):
            return None
        return run_problem(args, problem_id=problem_id, ...)

    for status in executor.map(run_one, problems):
        if status is None:
            continue
        append_status(args.status_jsonl, status)
```

### Pattern 3: PID Lock via fcntl Advisory Lock

**What:** Use POSIX `fcntl.flock(fd, LOCK_EX | LOCK_NB)` on a well-known file path to prevent concurrent instances of timing-sensitive scripts. The lock file contains the holding PID for diagnostics.

**When to use:** Any script that runs GPU timing (profiler or event-based) where concurrent GPU access would corrupt timing measurements.

**Trade-offs:**
- (+) Automatic cleanup on process death (kernel releases flock on fd close/process exit)
- (+) No stale lock files to clean up manually
- (+) Cross-platform POSIX support (works in Docker containers)
- (-) Only protects same-host, same-filesystem instances
- (-) Requires the lock file path to be writable

**Implementation sketch:**
```python
# src/sol_execbench/core/bench/pid_lock.py
import fcntl
import os
from pathlib import Path

class PidLock:
    def __init__(self, lock_path: Path):
        self._path = lock_path
        self._fd: int | None = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            holder = _read_holder_pid(self._path)
            os.close(self._fd)
            self._fd = None
            raise PidLockError(
                f"Another instance holds {self._path} "
                f"(PID: {holder or 'unknown'})"
            )
        os.ftruncate(self._fd, 0)
        os.write(self._fd, f"{os.getpid()}\n".encode())

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self):  self.acquire(); return self
    def __exit__(self, *exc): self.release()
```

### Pattern 4: Timing Isolation Audit

**What:** Pre-flight checks that detect GPU contention from other processes, validate clock lock state, and optionally probe L2 cache state before timing-sensitive execution begins.

**When to use:** Before the first profiling iteration in `run_rdna4_profiler_timing_batch.py` and before the first GPU evaluation in `run_dataset.py`.

**Trade-offs:**
- (+) Fails fast when the environment is contaminated
- (+) Can log warnings without blocking execution (advisory mode)
- (-) `rocm-smi` / `amd-smi` queries add latency to startup
- (-) L2 cache state is not directly queryable on RDNA4; check is heuristic-level only

**Implementation sketch:**
```python
# src/sol_execbench/core/bench/timing_isolation.py
def audit_timing_environment(
    *,
    gpu_id: int = 0,
    require_clock_locked: bool = False,
    check_gpu_contention: bool = True,
) -> TimingEnvironmentAudit:
    issues: list[str] = []
    # 1. Check clock lock state
    if require_clock_locked and not are_clocks_locked():
        issues.append("clock_unlocked")
    # 2. Check for other GPU processes via rocm-smi
    if check_gpu_contention:
        other_processes = _probe_gpu_processes(gpu_id)
        if other_processes:
            issues.append(f"gpu_contention: {len(other_processes)} other processes")
    # 3. Check SOL_EXECBENCH_CLOCKS_LOCKED env var consistency
    return TimingEnvironmentAudit(
        issues=issues,
        passed=len(issues) == 0,
    )
```

## Data Flow

### Current Flow: run_rdna4_profiler_timing_batch.py (serial)

```
_build_coverage() -> select_fallback_targets() -> [serial for target in targets]:
    _profile_target(target)       # CPU: packager, JSON parse
        collect_rocprofv3_timing()  # GPU: subprocess runs rocprofv3
    -> write replacement sidecar
-> _build_summary() -> write batch-summary.json
```

### Proposed Flow: run_rdna4_profiler_timing_batch.py (parallel prepare)

```
[NEW] pid_lock.acquire()
[NEW] audit_timing_environment()
_build_coverage() -> select_fallback_targets()
-> ThreadPoolExecutor: _prepare_profiler_target(target) for each target
-> serial: as_completed -> _profile_target(prepared) for each result
    -> write replacement sidecar
-> _build_summary() -> write batch-summary.json
[NEW] pid_lock.release()
```

### Current Flow: run_derived_isolated.py (serial for-loop)

```
discover_problems() -> load_completed() -> [serial for problem_dir in problems]:
    should_skip() -> run_problem() -> append_status()
```

### Proposed Flow: run_derived_isolated.py (parallel)

```
discover_problems() -> load_completed()
-> ThreadPoolExecutor: run_one(problem_dir) for each problem
-> append_status() in order (or per-completion with lock)
```

## Key Design Decisions

### 1. ThreadPoolExecutor over ProcessPoolExecutor

**Decision:** Use `ThreadPoolExecutor` for both scripts.
**Rationale:** The existing codebase uses `ThreadPoolExecutor` exclusively (run_dataset.py lines 42, 2417, 2514). Subprocess calls already provide process isolation for GPU work. CPU-bound work is I/O-heavy (JSON parsing, file staging) where GIL is released during I/O. There is no compute-bound Python parallelism that would benefit from separate processes.

### 2. fcntl flock over lockfile library

**Decision:** Use `fcntl.flock` directly instead of a third-party library.
**Rationale:** The project has zero existing file-locking dependencies. `fcntl` is in the Python standard library, automatic on process death, and sufficient for single-host Docker use. The `lockfile` library is unmaintained; `filelock` would add a dependency for trivial functionality.

### 3. Advisory-only isolation checks

**Decision:** Timing isolation checks should log warnings by default, not hard-fail.
**Rationale:** The existing `evaluation_stability.py` already records `clock_unlocked` and `profiler_overhead_risk` as reason codes without blocking execution. The new checks should follow the same advisory pattern: record in stability diagnostics, optionally fail if a `--strict-isolation` flag is set.

### 4. No changes to eval_driver.py or trace JSONL

**Decision:** All changes are script-layer and pre-flight. The subprocess eval driver, trace format, and sidecar schemas are untouched.
**Rationale:** Explicitly deferred by milestone scope: "Changing any public CLI interface, trace JSONL schema, or sidecar format" is out of scope.

## Anti-Patterns

### Anti-Pattern 1: Parallel GPU Execution

**What people might do:** Run multiple `rocprofv3` subprocesses concurrently to speed up batch timing.
**Why it is wrong:** ROCm's `rocprofv3` and GPU event timing assume exclusive GPU access. Concurrent kernel execution corrupts profiler timestamps, introduces L2 cache interference, and produces non-reproducible timing data. The milestone explicitly targets "CPU-parallel/GPU-serial."
**Do this instead:** Parallelize only CPU staging (packager construction, JSON parsing). Keep GPU subprocess execution strictly serial.

### Anti-Pattern 2: Shared Mutable Log File Without Synchronization

**What people might do:** Have multiple threads write to the same `--log-file` without synchronization in `run_derived_isolated.py`.
**Why it is wrong:** The log file is opened in append mode (line 214) and written to by multiple threads. Interleaved writes produce garbled log lines.
**Do this instead:** Use a `threading.Lock` around log writes, or buffer messages and flush serially per the `run_dataset.py` pattern (lines 2516-2518).

### Anti-Pattern 3: PID Lock with Stale File Detection

**What people might do:** Check if PID in lock file is still running, then steal the lock.
**Why it is wrong:** `fcntl.flock` is already released by the kernel when the holding process dies. Stale-file checking introduces a TOCTOU race between checking the PID and acquiring the lock.
**Do this instead:** Trust `fcntl.flock(LOCK_EX | LOCK_NB)`. If it fails, the lock is held. If the process died, the kernel already released it.

## Build Order

The dependency graph between new features determines the build order:

```
pid_lock.py (no deps)  ----+
                           |
timing_isolation.py        +--> run_rdna4_profiler_timing_batch.py modification
  depends on:              |        |
  clock_lock.py            |        v
  (existing)         ------+   [audit_timing_environment() call at start]
                           |
                           +--> run_derived_isolated.py modification
                           |        |
                           |        v
                           |   [ThreadPoolExecutor wrapper, pid_lock optional]
                           |
                           +--> evaluation_stability.py extension
                                    |
                                    v
                               [new reason codes: gpu_contention, multi_instance_risk]
```

### Recommended Build Order

1. **`pid_lock.py`** -- zero dependencies, testable in isolation with unit tests. Build first because both script modifications can import it.

2. **`timing_isolation.py`** -- depends only on existing `clock_lock.py` and system tools (`rocm-smi`, `amd-smi`). Testable without GPU via mocking.

3. **`run_derived_isolated.py` parallelism** -- simplest change: wrap the serial for-loop in `ThreadPoolExecutor.map`. No GPU concerns, no PID lock needed (derived phase is CPU-only). Lowest risk, highest immediate value.

4. **`run_rdna4_profiler_timing_batch.py` parallelism** -- the most complex change. Requires refactoring `_profile_target` into separate prepare (CPU) and execute (GPU) phases, adding `ThreadPoolExecutor` for prepare, adding PID lock acquisition, adding timing isolation audit.

5. **`evaluation_stability.py` extension** -- add new reason codes (`gpu_contention`, `multi_instance_interference`, `l2_cache_pollution_risk`) and wire timing isolation results into stability diagnostics.

6. **Integration tests** -- test PID lock contention (two processes), test parallel prepare with serial execute, test isolation audit output in stability reports.

## Modified vs New Components Summary

| Component | Action | Scope |
|-----------|--------|-------|
| `src/sol_execbench/core/bench/pid_lock.py` | **NEW** | PID lock module with fcntl, ~80 lines |
| `src/sol_execbench/core/bench/timing_isolation.py` | **NEW** | GPU contention detection, env audit, ~100 lines |
| `scripts/run_derived_isolated.py` | **MODIFY** | Add `--jobs` flag, wrap for-loop in ThreadPoolExecutor, ~30 lines changed |
| `scripts/run_rdna4_profiler_timing_batch.py` | **MODIFY** | Split _profile_target into prepare/execute, add ThreadPoolExecutor, add PID lock acquisition, ~60 lines changed |
| `src/sol_execbench/core/evaluation_stability.py` | **MODIFY** | Add reason codes for isolation issues, ~10 lines changed |
| `tests/sol_execbench/core/bench/test_pid_lock.py` | **NEW** | Unit tests for PID lock acquisition, contention, auto-release |
| `tests/sol_execbench/core/bench/test_timing_isolation.py` | **NEW** | Unit tests for isolation audit with mocked system tools |

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `pid_lock` <-> scripts | Import and context manager | Scripts acquire lock at entry, release at exit |
| `timing_isolation` <-> `run_rdna4_profiler_timing_batch.py` | Import and pre-flight call | Audit runs once before the target loop starts |
| `timing_isolation` <-> `evaluation_stability.py` | Shared reason code strings | New reason codes added to `STABILITY_STATUS_KEYS` |
| `timing_isolation` <-> `clock_lock.py` | Import `are_clocks_locked` | Reuses existing clock state detection |

### External Dependencies

| Dependency | Integration | Notes |
|------------|-------------|-------|
| `rocm-smi` | Subprocess call in `timing_isolation.py` | Used to detect other GPU processes; graceful degradation if unavailable |
| `amd-smi` | Subprocess call in `timing_isolation.py` | Used to verify clock state; already used by `clock_lock.py` |
| `fcntl` (stdlib) | Direct import in `pid_lock.py` | No new external dependencies |
| `concurrent.futures` (stdlib) | Direct import in both scripts | Already imported by `run_dataset.py` |

## Sources

- Direct codebase inspection of `scripts/run_dataset.py`, `scripts/run_rdna4_profiler_timing_batch.py`, `scripts/run_derived_isolated.py`
- Existing concurrency pattern in `run_dataset.py` lines 2417-2453 (pipeline mode) and 2487-2523 (derived parallel mode)
- Existing clock lock infrastructure in `src/sol_execbench/core/bench/clock_lock.py`
- Existing stability diagnostics in `src/sol_execbench/core/evaluation_stability.py`
- POSIX `fcntl(2)` flock semantics: kernel-managed, auto-released on process death

---
*Architecture research for: script parallelism and safety hardening*
*Researched: 2026-06-10*
