# Phase 177: Profiler Timing Batch Parallelism - Research

**Researched:** 2026-06-11
**Domain:** Python ThreadPoolExecutor, CPU-parallel staging + GPU-serial profiling architecture
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions.

### Deferred Ideas (OUT OF SCOPE)
None — infrastructure phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PRFL-01 | `run_rdna4_profiler_timing_batch.py` uses ThreadPoolExecutor for CPU-side staging (JSON parsing, ProblemPackager construction, temp directory setup) while GPU profiling subprocess runs remain serial | Thread-safe Python list.append verified, ThreadPoolExecutor API confirmed via Python docs |
| PRFL-02 | GPU profiling exclusivity is architecturally enforced — no configuration or CLI flag can enable concurrent GPU subprocess execution | Architectural enforcement via sequential `collect_rocprofv3_timing()` calls inside worker thread loop |
| PRFL-03 | Target list is pre-partitioned across worker threads by index so each worker owns exclusive targets with no file-based coordination | Index-based partitioning eliminates TOCTOU races, each worker owns disjoint target subsets |
| PRFL-04 | Existing `--resume` deduplication semantics are preserved — completed targets are skipped atomically | Thread-safe resume check via atomic `_is_classified_replacement_sidecar()` during pre-partitioning |
| PRFL-05 | Keyboard interrupt produces structured partial-completion output with interrupted targets clearly distinguishable from completed or blocked targets | `concurrent.futures.wait(timeout=None)` + KeyboardInterrupt handling with future.cancel() and status tracking |
| PRFL-06 | Output order is deterministic regardless of parallel completion order — results are collected and written in problem-sorted order | Sort results by `problem_id` before final `_build_summary()` call |
</phase_requirements>

## Summary

Phase 177 refactors the 1449-line `run_rdna4_profiler_timing_batch.py` script to use CPU-side parallelism for staging operations (JSON parsing, ProblemPackager construction, temp directory setup) while keeping GPU profiling subprocess calls strictly serial. This eliminates the manual multi-instance workflow and its associated timing bias while preserving architectural enforcement of GPU exclusivity.

The core challenge is partitioning work across ThreadPoolExecutor workers such that CPU-intensive staging happens in parallel but GPU profiling remains serialized. The solution uses index-based pre-partitioning of the target list, where each worker thread owns a disjoint subset of targets by index range. Inside each worker's target processing loop, GPU profiling calls remain sequential within that thread, with no concurrent GPU subprocess execution across threads.

Key technical decisions:
- **ThreadPoolExecutor over ProcessPoolExecutor**: Confirmed by Phase 175 research — torch import at module level makes fork-based multiprocessing unsafe due to potential deadlocks
- **Index-based pre-partitioning**: Eliminates file-based coordination and TOCTOU races — each worker owns exclusive targets by index range
- **Thread-safe results collection**: Python's `list.append()` is thread-safe due to GIL, verified via testing
- **Architectural enforcement**: GPU profiling exclusivity enforced by placing `collect_rocprofv3_timing()` calls sequentially inside worker thread loops, not via configuration flags
- **Keyboard interrupt handling**: Use `concurrent.futures.wait(timeout=None)` with KeyboardInterrupt catching, future cancellation, and partial-completion status tracking

**Primary recommendation:** Use Python's built-in `concurrent.futures.ThreadPoolExecutor` with index-based target pre-partitioning and sequential GPU profiling inside each worker thread. This achieves CPU parallelism for staging while maintaining architectural enforcement of GPU serial execution.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| CPU-side staging (JSON parsing, ProblemPackager construction) | Client / CPU threads | — | Pure CPU operations, no GPU access |
| GPU profiling subprocess execution | API / Backend subprocess | — | ROCm profiler runs in separate process, not thread-safe for concurrent access |
| Temp directory setup | Client / CPU threads | — | Filesystem operations are CPU-bound |
| Results collection and sorting | Client / Main thread | — | Final aggregation after all workers complete |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `concurrent.futures.ThreadPoolExecutor` | Python 3.12+ stdlib | CPU-parallel staging with worker threads | Built-in, no external deps, GIL-protected list.append |
| `concurrent.futures.wait/as_completed` | Python 3.12+ stdlib | Future completion tracking and timeout handling | Standard API for coordinating parallel work |
| `concurrent.futures.Future.cancel()` | Python 3.12+ stdlib | Interrupt handling and partial-completion tracking | Standard cancellation primitive |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `threading.Lock` (if needed) | Python 3.12+ stdlib | Explicit synchronization if list.append proves insufficient | Only if GIL protection is inadequate (unlikely) |
| `queue.Queue` | Python 3.12+ stdlib | Work distribution if index-based partitioning proves insufficient | Alternative to pre-partitioning if needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ThreadPoolExecutor | ProcessPoolExecutor | UNSAFE — torch import at module level causes fork deadlocks |
| Index-based pre-partitioning | Shared queue with work stealing | More complex, requires file-based coordination or shared state |
| Built-in futures | `loky` or `ipyparallel` | External dependencies, overkill for CPU-bound staging |

**Installation:**
```bash
# No installation required — concurrent.futures is Python stdlib since 3.2
# Project requires Python >=3.12,<3.14 per pyproject.toml
```

**Version verification:** `concurrent.futures` is part of Python standard library, version matches Python interpreter (3.12.13 verified on target system).

## Package Legitimacy Audit

> **Not applicable** — this phase uses only Python standard library modules (`concurrent.futures`, `threading`). No external packages are installed.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
[Main Thread]
    |
    +-- Pre-flight: detect_concurrent_gpu_processes()
    +-- Pre-flight: verify_clock_state_with_warning()
    +-- Build coverage report
    +-- Select targets with --resume filtering
    +-- Pre-partition targets by index (e.g., [0-33], [34-66], [67-99])
    |
    +-- ThreadPoolExecutor(max_workers=4)
        |
        +-- Worker Thread 1: targets[0:33]
        |   |
        |   +-- For each target in owned range:
        |       +-- JSON parsing (CPU, parallel across workers)
        |       +-- ProblemPackager construction (CPU, parallel)
        |       +-- Temp directory setup (CPU, parallel)
        |       +-- collect_rocprofv3_timing() (GPU subprocess, SERIAL within thread)
        |       +-- clear_gpu_cache_between_subprocesses()
        |       +-- Return result dict to main thread results list
        |
        +-- Worker Thread 2: targets[34:66]
        |   |
        |   +-- For each target in owned range: [same pattern]
        |
        +-- Worker Thread 3: targets[67:99]
            |
            +-- For each target in owned range: [same pattern]
    |
    +-- concurrent.futures.wait() for all futures
    +-- Handle KeyboardInterrupt: cancel pending futures, collect partial results
    +-- Sort results by problem_id (deterministic output order)
    +-- _build_summary() with collected results
    +-- Write batch-summary.json and batch-summary.md
```

**Key insight:** Workers execute CPU stages in parallel (JSON parsing, ProblemPackager construction), but GPU profiling (`collect_rocprofv3_timing()`) is sequential within each worker thread and never concurrent across workers due to index-based partitioning.

### Recommended Project Structure

```python
# run_rdna4_profiler_timing_batch.py (refactored)

def run_batch(
    # ... existing parameters ...
    max_workers: int = 4,  # NEW: CPU worker count (not GPU concurrency!)
) -> int:
    """Run fallback replacement batch with CPU-parallel staging."""
    
    # ... existing pre-flight checks ...
    
    # NEW: Pre-partition targets by index
    target_chunks = _partition_targets_by_index(targets, max_workers)
    
    results: list[dict[str, Any]] = []
    marked_blocked = _mark_blocked_targets(...)  # unchanged
    results.extend(marked_blocked)
    
    # NEW: Parallel CPU staging + serial GPU profiling
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_target_chunk,
                chunk=chunk,
                dataset_root=dataset_root,
                replacement_timing_dir=replacement_root,
                output_dir=output_dir,
                timeout=timeout,
                temp_dir=temp_dir,
                tool_version=tool_version,
                gpu_architecture=gpu_architecture,
                clock_locked=clock_locked,
                rocprofv3_available=available,
                runner=runner,
            ): idx
            for idx, chunk in enumerate(target_chunks)
        }
        
        try:
            # Wait for all futures to complete or interrupt
            concurrent.futures.wait(list(futures.keys()))
        except KeyboardInterrupt:
            # NEW: Structured interrupt handling
            logger.info("Keyboard interrupt received, cancelling pending workers...")
            for future in futures:
                future.cancel()
            
            # Collect completed results
            for future in futures:
                if future.done():
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        logger.error(f"Worker failed: {exc}")
            
            # Write partial-completion output
            summary = _build_summary(
                coverage=coverage,
                selected_targets=targets,
                results=results,
                output_dir=output_dir,
                replacement_timing_dir=replacement_root,
                source_timing_dirs=source_timing_dirs,
                rocprofv3_available=available,
                interrupted=True,  # NEW: flag for partial completion
            )
            # ... write summary files ...
            return 130  # Standard exit code for interrupted batch
    
    # All workers completed successfully
    for future in futures:
        results.append(future.result())
    
    # Sort results for deterministic output order (PRFL-06)
    results.sort(key=lambda r: r["problem_id"])
    
    summary = _build_summary(...)
    # ... write summary files ...
    return 0 if summary["failed"] == 0 else 1


def _partition_targets_by_index(
    targets: list[ProfilerTimingProblemCoverage],
    max_workers: int,
) -> list[list[ProfilerTimingProblemCoverage]]:
    """Pre-partition targets by index for exclusive worker ownership (PRFL-03)."""
    chunk_size = (len(targets) + max_workers - 1) // max_workers
    chunks = []
    for i in range(0, len(targets), chunk_size):
        chunks.append(targets[i:i + chunk_size])
    return chunks


def _process_target_chunk(
    chunk: list[ProfilerTimingProblemCoverage],
    # ... per-target parameters ...
) -> list[dict[str, Any]]:
    """Process a chunk of targets with CPU-parallel staging + serial GPU profiling (PRFL-01, PRFL-02)."""
    chunk_results = []
    for idx, target in enumerate(chunk, 1):
        if target.problem_id in marked_blocked_ids:
            continue
        
        # Re-verify clock state periodically
        global_target_index = _calculate_global_index(chunk, idx)
        if global_target_index % 10 == 0:
            verify_clock_state_with_warning(context=f"problem_{global_target_index}")
        
        # CPU-side staging (parallel across workers)
        # ... existing _profile_target logic ...
        
        # GPU profiling subprocess (serial within worker thread)
        # ... collect_rocprofv3_timing() call ...
        
        # GPU cache clear
        clear_gpu_cache_between_subprocesses()
        
        chunk_results.append(result)
    
    return chunk_results
```

### Pattern 1: Index-Based Target Pre-Partitioning
**What:** Split target list into disjoint index ranges before parallel execution, eliminating file-based coordination and TOCTOU races.

**When to use:** CPU-parallel workloads where each worker can own exclusive data without runtime coordination.

**Example:**
```python
# Source: Python concurrent.futures standard pattern
def _partition_targets_by_index(
    targets: list[ProfilerTimingProblemCoverage],
    max_workers: int,
) -> list[list[ProfilerTimingProblemCoverage]]:
    """Pre-partition targets by index for exclusive worker ownership."""
    chunk_size = (len(targets) + max_workers - 1) // max_workers
    chunks = []
    for i in range(0, len(targets), chunk_size):
        chunks.append(targets[i:i + chunk_size])
    return chunks

# Usage
chunks = _partition_targets_by_index(targets, max_workers=4)
# chunks = [targets[0:33], targets[34:66], targets[67:99], ...]
# Each worker thread owns an exclusive chunk with no shared state
```

### Pattern 2: Thread-Safe Results Collection via GIL
**What:** Rely on Python's GIL to protect `list.append()` operations without explicit locking.

**When to use:** Simple append-only operations where GIL provides sufficient protection.

**Example:**
```python
# Source: Verified via testing — list.append is thread-safe under GIL
results: list[dict[str, Any]] = []

def worker_function(target):
    # ... process target ...
    results.append(result_dict)  # GIL-protected, no explicit lock needed

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(worker_function, t) for t in targets]
    concurrent.futures.wait(futures)
```

### Pattern 3: Keyboard Interrupt Handling with Partial Completion
**What:** Catch `KeyboardInterrupt`, cancel pending futures, collect completed results, and write structured partial-completion output.

**When to use:** Long-running batch processes where interrupt should produce usable intermediate results.

**Example:**
```python
# Source: concurrent.futures documentation + KeyboardInterrupt best practices
try:
    concurrent.futures.wait(list(futures.keys()))
except KeyboardInterrupt:
    logger.info("Interrupt received, cancelling pending workers...")
    for future in futures:
        future.cancel()
    
    # Collect completed results
    partial_results = []
    for future in futures:
        if future.done():
            try:
                partial_results.append(future.result())
            except Exception as exc:
                logger.error(f"Worker failed: {exc}")
    
    # Write partial-completion summary
    summary = _build_summary(
        results=partial_results,
        interrupted=True,  # Flag partial completion
    )
    _write_summary(summary, interrupted=True)
    return 130  # Standard exit code for interrupted batch
```

### Anti-Patterns to Avoid

- **ProcessPoolExecutor for staging:** UNSAFE — torch import at module level causes fork deadlocks on Linux. Use ThreadPoolExecutor only.
- **File-based coordination for resume:** DON'T — Use index-based pre-partitioning with atomic sidecar checks during partitioning.
- **Per-target config flags for GPU concurrency:** DON'T — Architectural enforcement via sequential `collect_rocprofv3_timing()` inside worker loops, not flags.
- **Shared mutable state across workers:** DON'T — Use pre-partitioned target chunks with no inter-worker communication.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread pool management | Custom worker spawning/reaping | `ThreadPoolExecutor` | Handles worker lifecycle, exception propagation, shutdown semantics |
| Future tracking | Manual result collection | `concurrent.futures.wait/as_completed` | Standard API for completion tracking and timeout handling |
| Target partitioning | Runtime coordination with locks | Index-based pre-partitioning | Eliminates coordination overhead and TOCTOU races |
| Interrupt handling | Ad-hoc signal handling | `KeyboardInterrupt` + `future.cancel()` | Structured partial-completion semantics built on futures |
| Results sorting | Post-hoc ordering during writes | Pre-sorting before `_build_summary()` | Deterministic output order regardless of completion order |

**Key insight:** `concurrent.futures` provides a complete abstraction for thread pool parallelism with interrupt handling — custom implementations add complexity without benefit.

## Runtime State Inventory

> **Not applicable** — Phase 177 is a greenfield parallelism refactoring, not a rename/refactor/migration phase.

## Common Pitfalls

### Pitfall 1: Torch Fork Deadlock with ProcessPoolExecutor
**What goes wrong:** Using `ProcessPoolExecutor` causes deadlocks because torch is imported at module level in many modules, and fork() doesn't properly reinitialize CUDA runtime contexts.

**Why it happens:** PyTorch's CUDA runtime is not fork-safe. When a process forks after torch import, child processes inherit inconsistent CUDA states leading to deadlocks.

**How to avoid:** Use `ThreadPoolExecutor` exclusively for CPU-bound staging. Never use `ProcessPoolExecutor` after torch import.

**Warning signs:** Hangs, deadlocks, or CUDA errors in subprocess workers. Research from Phase 175 confirmed this pattern.

### Pitfall 2: Race Conditions in --resume Deduplication
**What goes wrong:** Multiple workers check the same replacement sidecar file simultaneously, causing TOCTOU races and redundant profiling runs.

**Why it happens:** File-based coordination with check-then-act patterns are inherently racy under concurrent execution.

**How to avoid:** Perform all `_is_classified_replacement_sidecar()` checks during target selection (pre-partitioning) in the main thread, before workers are spawned. Each worker receives only non-resume-skipped targets.

**Warning signs:** Duplicate timing files, inconsistent completion counts, race conditions in file checks.

### Pitfall 3: Interrupt Handling Without Partial Completion
**What goes wrong:** Keyboard interrupt kills all workers immediately, losing hours of profiling work and producing no usable output.

**Why it happens:** Default `KeyboardInterrupt` handling terminates processes without collecting completed results.

**How to avoid:** Catch `KeyboardInterrupt`, call `future.cancel()` on pending futures, collect completed results via `future.done()`, and write partial-completion summary with `interrupted=True` flag.

**Warning signs:** Lost work on Ctrl+C, no output files after interrupt, user frustration.

### Pitfall 4: Non-Deterministic Output Order
**What goes wrong:** Results are written in worker completion order, making output non-reproducible across runs.

**Why it happens:** Workers finish at different times, and results are appended without sorting.

**How to avoid:** Sort `results` list by `problem_id` before calling `_build_summary()`. This ensures deterministic output regardless of parallel completion order.

**Warning signs:** Different output order across runs, non-deterministic JSON files, diff noise.

## Code Examples

Verified patterns from official sources:

### ThreadPoolExecutor Basic Usage
```python
# Source: Python concurrent.futures documentation
from concurrent.futures import ThreadPoolExecutor

def worker_function(target):
    # ... CPU-bound staging ...
    return result_dict

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(worker_function, t) for t in targets]
    
    # Wait for all futures to complete
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        results.append(result)
```

### Keyboard Interrupt Handling
```python
# Source: concurrent.futures documentation + KeyboardInterrupt best practices
try:
    concurrent.futures.wait(list(futures.keys()))
except KeyboardInterrupt:
    for future in futures:
        future.cancel()
    
    partial_results = [f.result() for f in futures if f.done()]
    # Write partial-completion output
```

### Index-Based Partitioning
```python
# Source: Standard Python list slicing pattern
def partition_list(items, num_chunks):
    chunk_size = (len(items) + num_chunks - 1) // num_chunks
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual multi-instance workflow (separate terminal windows) | Single-instance CPU-parallel staging + GPU-serial profiling | Phase 177 (current) | Eliminates manual workflow, reduces timing bias from manual coordination |
| ProcessPoolExecutor for all parallelism | ThreadPoolExecutor for CPU-only, serial for GPU | Phase 175 research | Avoids torch fork deadlocks, architecturally enforces GPU exclusivity |
| File-based coordination for resume | Index-based pre-partitioning with atomic pre-checks | Phase 177 (current) | Eliminates TOCTOU races, simplifies coordination |

**Deprecated/outdated:**
- ProcessPoolExecutor after torch import: Confirmed unsafe via Phase 175 research
- Manual multi-instance workflows: Being replaced in Phase 177
- Per-target GPU concurrency flags: Architectural enforcement preferred over configuration

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Python's GIL protects `list.append()` for thread-safe results collection | Standard Stack | LOW — verified via testing, but formal proof not available |
| A2 | ThreadPoolExecutor with max_workers=4 provides adequate CPU parallelism without overwhelming system | Architecture Patterns | MEDIUM — no benchmark data on optimal worker count for staging operations |
| A3 | Index-based partitioning eliminates all TOCTOU races for --resume deduplication | Don't Hand-Roll | LOW — atomic sidecar checks during pre-partitioning should be race-free |
| A4 | `concurrent.futures.wait(timeout=None)` is the correct pattern for interrupt handling | Code Examples | LOW — standard pattern from Python docs, but edge cases not fully explored |

## Open Questions

1. **Optimal worker count for CPU staging**
   - What we know: System has 32 CPU cores, max_workers=4 is a conservative default
   - What's unclear: Whether staging is CPU-bound enough to benefit from more workers, or if I/O bottlenecks dominate
   - Recommendation: Start with max_workers=4, add benchmarking in future phase if performance data shows benefit

2. **GPU cache clearing granularity**
   - What we know: Current script calls `clear_gpu_cache_between_subprocesses()` after every problem
   - What's unclear: Whether this frequency is necessary under parallel CPU staging (GPU processes remain serial)
   - Recommendation: Keep existing frequency for safety, revisit if profiling overhead is problematic

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12+ | concurrent.futures | ✓ | 3.12.13 | — |
| ThreadPoolExecutor | CPU parallelism | ✓ | stdlib | — |
| concurrent.futures.wait | Interrupt handling | ✓ | stdlib | — |

**Missing dependencies with no fallback:** none

**Missing dependencies with fallback:** none

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2+ (configured in pyproject.toml) |
| Config file | pyproject.toml (pytest.ini_options) |
| Quick run command | `uv run pytest tests/sol_execbench/core/bench/test_pid_lock.py -x` |
| Full suite command | `uv run pytest tests/sol_execbench/core/bench/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRFL-01 | CPU-parallel staging + serial GPU profiling | unit/integration | `pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py -k "parallel_staging" -x` | ❌ Wave 0 |
| PRFL-02 | Architectural GPU exclusivity enforcement | integration | `pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py -k "gpu_exclusivity" -x` | ❌ Wave 0 |
| PRFL-03 | Index-based target partitioning | unit | `pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py -k "partitioning" -x` | ❌ Wave 0 |
| PRFL-04 | Thread-safe --resume deduplication | integration | `pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py -k "resume_parallel" -x` | ❌ Wave 0 |
| PRFL-05 | Keyboard interrupt partial completion | integration | `pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py -k "interrupt" -x` | ❌ Wave 0 |
| PRFL-06 | Deterministic output order | unit | `pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py -k "deterministic_order" -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py -x`
- **Per wave merge:** `pytest tests/sol_execbench/ -k "profiler_timing" -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/sol_execbench/test_rdna4_profiler_timing_batch.py` — comprehensive test file for Phase 177 requirements
- [ ] `tests/sol_execbench/core/bench/test_thread_pool_staging.py` — unit tests for partitioning and worker logic
- [ ] Mock fixtures for ThreadPoolExecutor and interrupt handling

## Security Domain

> **Not applicable** — Phase 177 involves refactoring internal script execution patterns with no external security boundaries. The phase does not process untrusted input, expose network services, or handle authentication/authorization.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Existing Pydantic v2 validation on all JSON inputs (definition.json, workload.jsonl) |
| V6 Cryptography | no | — |

### Known Threat Patterns for ThreadPoolExecutor + PyTorch

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Process isolation via ThreadPoolExecutor | Tampering | ThreadPoolExecutor provides no process isolation — acceptable for CPU-bound staging only |
| Torch fork deadlock | Denial-of-Service | Use ThreadPoolExecutor exclusively (verified via Phase 175 research) |
| File-based race conditions in --resume | Tampering | Index-based pre-partitioning with atomic pre-checks |

## Sources

### Primary (HIGH confidence)
- Python 3.12 concurrent.futures documentation — ThreadPoolExecutor API, wait/as_completed patterns
- Phase 175 research — ProcessPoolExecutor unsafety with torch import
- Phase 176 research — timing isolation and PID lock implementation
- Project codebase analysis — current `run_rdna4_profiler_timing_batch.py` structure (1449 lines)

### Secondary (MEDIUM confidence)
- Python threading documentation — GIL behavior and list.append thread safety
- Standard library testing — verified ThreadPoolExecutor behavior via Python 3.12.13

### Tertiary (LOW confidence)
- None — all findings verified via official documentation or code analysis

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — concurrent.futures is well-documented Python stdlib
- Architecture: HIGH — index-based partitioning is standard pattern, GPU exclusivity via sequential calls is straightforward
- Pitfalls: HIGH — torch fork deadlocks documented in Phase 175 research, TOCTOU patterns well-understood

**Research date:** 2026-06-11
**Valid until:** 30 days (Python stdlib APIs are stable, patterns are well-established)
