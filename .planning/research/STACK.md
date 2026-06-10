# Project Research - Stack for Script Parallelism, PID Locking, and Benchmark Reproducibility

## Scope

This research covers stack additions and integration points needed only for
v1.35 milestone features: internal CPU-parallel/GPU-serial concurrency in two
Python CLI scripts, PID-based multi-instance prevention, and benchmark timing
environment isolation/reproducibility hardening. It deliberately does not
re-research existing project capabilities (subprocess.run, ThreadPoolExecutor
in run_dataset.py, rocprofv3 profiling, file-based --resume deduplication).

**Researched:** 2026-06-10

---

## Recommended Stack

### 1. CPU-Parallel Work Scheduling with GPU-Serial Enforcement

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `concurrent.futures.ThreadPoolExecutor` | Python stdlib (3.12+) | CPU-stage parallel work dispatch | Already in use in `run_dataset.py` for derived phase and pipeline mode. The GIL is not a concern here because CPU-stage work is I/O-bound (file staging, JSON parsing, directory setup) or releases the GIL (subprocess.run). ThreadPoolExecutor avoids the pickle/IPC overhead of ProcessPoolExecutor and integrates cleanly with the existing `as_completed` pattern. |
| `concurrent.futures.as_completed` | Python stdlib (3.12+) | Collect results as GPU subprocesses finish | Already used in `run_dataset.py` pipeline mode. Preserves per-problem ordering semantics needed for resume/checkpoint. |
| `threading.Semaphore` | Python stdlib (3.12+) | GPU-serial enforcement gate | A semaphore initialized to 1 acts as a mutex controlling exclusive GPU access. CPU-side workers acquire the semaphore only for the GPU subprocess invocation, releasing it immediately after subprocess completion. This is simpler and more composable than a dedicated GPU worker thread, and avoids the complexity of a producer/consumer queue split. |
| `threading.Thread` | Python stdlib (3.12+) | Optional: dedicated GPU runner thread | Alternative to the semaphore approach: one thread drains a `queue.Queue` of prepared GPU work items. This provides cleaner separation but adds one extra layer of indirection. Recommended only if the semaphore proves awkward. |

**Integration pattern for `run_rdna4_profiler_timing_batch.py`:**

```
1. Main thread: build coverage, select targets
2. ThreadPoolExecutor(max_workers=N):
   - Each worker does CPU prep (load definition, build packager, stage files)
   - Worker acquires GPU semaphore
   - Worker runs subprocess (rocprofv3 collection) -- this is GPU-serial
   - Worker releases GPU semaphore
   - Worker writes sidecar result (CPU I/O)
3. Main thread: collect futures, build summary
```

**Integration pattern for `run_derived_isolated.py`:**

```
1. Main thread: discover problems, filter completed
2. ThreadPoolExecutor(max_workers=N):
   - Each worker runs subprocess for one problem
   - These are CPU-only (derived sidecar generation), no GPU semaphore needed
   - Worker appends status to JSONL (with append-safe file write)
3. Main thread: summary
```

**Why NOT ProcessPoolExecutor:**
- The scripts already use `subprocess.run` for actual GPU work, so the GIL is released during GPU subprocess execution.
- ProcessPoolExecutor would require all callables and return values to be picklable, which is fragile with the current Pydantic model and Path-heavy data flow.
- Each worker process would need its own PyTorch import (~1.5 GB RSS per process), making memory usage prohibitive (as noted in the project's own pytest configuration comments).

**Why NOT asyncio:**
- The scripts use blocking `subprocess.run` calls with `capture_output=True`. Converting to asyncio would require `asyncio.create_subprocess_exec` and rewriting the entire flow as coroutines.
- The team already has ThreadPoolExecutor patterns in the codebase. Introducing asyncio would be a paradigm shift with no performance benefit for this I/O + subprocess-bound workload.

### 2. PID-Based Multi-Instance Prevention

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `fcntl.flock` (Linux stdlib) | Python stdlib (3.12+) | Non-blocking exclusive file lock | This project is Linux-only (ROCm). `fcntl.flock(LOCK_EX | LOCK_NB)` provides kernel-level advisory locking that is automatically released when the process exits (even on crash/OOM-kill). No external dependency needed. The lock is per-file-descriptor, so pointing it at a `.lock` file in the output directory gives natural per-output-directory isolation. |
| `os.getpid()` | Python stdlib (3.12+) | Record PID in lockfile metadata | Writing the PID into the lockfile allows diagnostic messages ("locked by PID 12345") and stale-lock investigation without relying on `fuser` or `lsof`. |

**Why NOT `filelock` (PyPI package):**
- `filelock` 3.29.0 is well-maintained (tox-dev, MIT license, supports Python 3.10-3.14) and provides cross-platform locking. However, this project is explicitly Linux-only, so the cross-platform benefit is wasted.
- `fcntl.flock` is zero-dependency, kernel-level, and has automatic release-on-crash semantics. `filelock` internally uses `fcntl.flock` on Linux anyway, so there is no functional advantage.
- Adding a dependency for something the stdlib already does perfectly on the only supported platform is unjustified.

**Why NOT lockfile (older PyPI package):**
- Stale lock handling (PID file checks) adds complexity. `fcntl.flock` with `LOCK_NB` avoids stale locks entirely because the kernel releases the lock when the process dies.

**Recommended implementation:**

```python
import fcntl
import os
import sys
from pathlib import Path

class ProcessLock:
    """Non-blocking exclusive process lock using fcntl.flock."""

    def __init__(self, lock_path: Path, *, name: str = "script") -> None:
        self._lock_path = lock_path
        self._name = name
        self._fd: int | None = None

    def acquire(self) -> bool:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            # Read existing PID for diagnostic message
            try:
                existing = os.read(fd, 64).decode().strip()
            except OSError:
                existing = "<unknown>"
            os.close(fd)
            print(
                f"ERROR: Another {self._name} instance is running "
                f"(lock: {self._lock_path}, holder: {existing}).",
                file=sys.stderr,
            )
            return False
        os.write(fd, f"pid={os.getpid()}\n".encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "ProcessLock":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
```

**Scripts that need locking:**
- `run_rdna4_profiler_timing_batch.py` -- exclusive GPU access with rocprofv3
- `run_derived_isolated.py` -- shared output directory writes (unless parallelized)
- `run_dataset.py` -- when using GPU phases (already has resume dedup, but concurrent instances can corrupt output)

**Lock file location convention:** `<output_dir>/.<script_name>.pid.lock`

### 3. Benchmark Timing Environment Isolation and Reproducibility

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `torch.cuda.synchronize()` | PyTorch ROCm (2.10.0+) | GPU-side synchronization barrier | Already used implicitly through HIP event timing. Explicit calls between warmup/timing phases ensure the GPU pipeline is drained before timing starts. The project's timing code in `src/sol_execbench/core/bench/timing.py` already uses PyTorch device events which include implicit synchronization. |
| `amd-smi set -l STABLE_PEAK` | ROCm tooling | GPU clock frequency locking | Already implemented in `src/sol_execbench/core/bench/clock_lock.py`. The v1.35 milestone should audit that clock-lock verification runs before each timing batch, not just at script start. |
| `torch.cuda.empty_cache()` | PyTorch ROCm (2.10.0+) | Clear PyTorch CUDA memory cache between problems | Prevents L2 cache pollution from one problem's tensors affecting the next problem's timing. Should be called after each problem's evaluation completes, before the next problem's staging begins. |
| Environment variable isolation | Python stdlib `os.environ` | Clean subprocess environment | Each subprocess invocation should start from a minimal environment rather than inheriting the full parent environment. Key variables to control: `CUDA_VISIBLE_DEVICES` (maps to ROCm devices), `HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_ROCM_ARCH`, `ROCprofilerSDK_*`. |
| Staging directory isolation | `tempfile.mkdtemp` (already in use) | Per-problem temp directories | Already used in `_profile_target` via `tempfile.mkdtemp(prefix="sol_execbench_rdna4_prof_batch_")`. The audit should verify these are cleaned up after each problem, not accumulated. |

**Why NOT GPU process isolation (separate processes per problem):**
- The scripts already use subprocess-per-problem for GPU evaluation. The concern is about what happens *within* the subprocess. The eval_driver.py template runs warmup, correctness, and timing in sequence within one process.
- Adding a process barrier *between* problems in a batch is already achieved by the subprocess boundary.

**Reproducibility audit checklist (not a library, but a process):**

1. **Clock lock verification:** Before batch start, verify `amd-smi set -l STABLE_PEAK` succeeded. After each N problems, re-verify clock frequency is stable.
2. **Memory isolation:** After each problem subprocess, call `torch.cuda.empty_cache()` in the parent process (if PyTorch is loaded) or rely on subprocess boundary for isolation.
3. **Environment consistency:** Record environment snapshot (ROCm version, PyTorch version, GPU arch) at batch start and include in batch summary.
4. **Staging cleanup:** Verify `tempfile.mkdtemp` staging directories are removed after each problem, or at batch end.
5. **L2 cache warming:** The eval_driver already runs `warmup_runs` iterations. The audit should verify this is sufficient and consistent.

---

## What NOT to Add

| Category | Rejected Option | Why |
|----------|----------------|-----|
| Concurrency | `asyncio` | Blocking subprocess.run calls, existing ThreadPoolExecutor patterns, no performance benefit |
| Concurrency | `multiprocessing.Pool` / `ProcessPoolExecutor` | Pickle constraints, 1.5 GB RSS per worker from PyTorch, GIL released during subprocess.run anyway |
| Concurrency | `joblib` / `dask` | External dependencies for something stdlib handles; designed for CPU-bound numpy work, not subprocess orchestration |
| Locking | `filelock` PyPI package | Linux-only project; stdlib fcntl.flock does the same thing with zero dependencies |
| Locking | `portalocker` | Same reasoning as filelock; unnecessary dependency |
| Locking | PID file with manual stale detection | fcntl.flock auto-releases on process death; stale detection is error-prone |
| Timing | `psutil` for process isolation | Heavy dependency; subprocess boundaries already provide isolation |
| Timing | Custom GPU cache flusher | torch.cuda.empty_cache() + subprocess boundary is sufficient |
| Timing | `numactl` / CPU pinning | Out of scope for this milestone; would require per-hardware tuning |

---

## Installation

No new package installation required. All recommended technologies are Python stdlib or already available through PyTorch ROCm.

```bash
# Nothing to install. All capabilities come from:
# - Python 3.12+ stdlib: concurrent.futures, fcntl, threading, os
# - PyTorch ROCm 2.10.0+: torch.cuda.synchronize(), torch.cuda.empty_cache()
# - ROCm tooling: amd-smi (already in clock_lock.py)
```

---

## Integration Points

### `run_rdna4_profiler_timing_batch.py`

**Current state:** Serial `for target in targets` loop calling `_profile_target` or `_profile_target_workload_sharded`.

**Changes needed:**
1. Add `ProcessLock` at script entry using `<output_dir>/.profiler_timing_batch.pid.lock`.
2. Wrap the target loop in `ThreadPoolExecutor(max_workers=cpu_workers)` where `cpu_workers` is configurable via `--prepare-jobs` (matching the naming convention from `run_dataset.py`).
3. Add a `threading.Semaphore(1)` for GPU access. Each worker acquires the semaphore before calling `_profile_target`, releases after subprocess completes.
4. Move CPU-only work (coverage building, target selection, sidecar writing) outside the semaphore.
5. `--resume` logic remains unchanged (file-based deduplication is thread-safe because each problem writes to a unique path).

### `run_derived_isolated.py`

**Current state:** Serial `for problem_dir in problems` loop calling `run_problem` via `subprocess.run`.

**Changes needed:**
1. Add `ProcessLock` at script entry using `<output_dir>/.derived_isolated.pid.lock`.
2. Wrap the problem loop in `ThreadPoolExecutor(max_workers=N)` where N is configurable via `--jobs` (matching existing convention).
3. No GPU semaphore needed -- derived phase is CPU-only sidecar generation.
4. `append_status` needs thread-safety: use a `threading.Lock` around the file append, or collect results and write once after all futures complete.
5. `--resume` logic remains unchanged (JSONL status file is append-only with idempotent completion check).

### `run_dataset.py`

**Current state:** Already has `ThreadPoolExecutor` for derived phase and pipeline mode. No changes needed for parallelism, but should receive `ProcessLock` for GPU phases.

**Changes needed:**
1. Add `ProcessLock` when `--phase` is `traces`, `all`, or `timing` (GPU-touching phases). Lock path: `<output_dir>/.dataset_run.pid.lock`.
2. Not needed for `--phase derived` (CPU-only).

### Environment Audit Points

These are not new libraries but code-level checks to add:

1. In `_profile_target`: verify staging dir cleanup (`shutil.rmtree` or `tempfile.TemporaryDirectory` context manager).
2. In batch summary: record `torch.version.hip`, `torch.cuda.get_device_name()`, `rocm-smi --showclocks` output.
3. In eval_driver template: verify `torch.cuda.synchronize()` is called between warmup and timing iterations.
4. In clock lock module: add periodic re-verification during long batches (every N problems).

---

## Sources

- Python stdlib `concurrent.futures` documentation (Python 3.12+, verified 2026-06-10): https://docs.python.org/3/library/concurrent.futures.html
- Python stdlib `fcntl.flock` documentation (Linux, available on this system): verified via `help(fcntl.flock)`
- PyPI `filelock` 3.29.0 (researched and rejected): https://pypi.org/project/filelock/
- Project's existing ThreadPoolExecutor usage in `scripts/run_dataset.py` (lines 2417-2453 for pipeline mode, lines 2514-2523 for derived parallel)
- Project's existing subprocess pattern in `scripts/run_rdna4_profiler_timing_batch.py` (`_staging_runner` function, line 1131-1148)
- Project's existing clock lock implementation in `src/sol_execbench/core/bench/clock_lock.py`
- Project's evaluation stability module in `src/sol_execbench/core/evaluation_stability.py`
- Project's pyproject.toml: `requires-python >= 3.12, < 3.14`
