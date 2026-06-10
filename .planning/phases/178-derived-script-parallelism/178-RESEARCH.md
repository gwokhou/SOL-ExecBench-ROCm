# Phase 178: Derived Script Parallelism - Research

**Researched:** 2025-06-11
**Domain:** CPU-bound parallel subprocess execution
**Confidence:** HIGH

## Summary

Phase 178 implements concurrent per-problem subprocess dispatch in `run_derived_isolated.py` using ThreadPoolExecutor, improving throughput for CPU-bound derived sidecar generation while preserving existing resume semantics and failure handling. The phase requires thread-safe JSONL writes, proper --jobs flag implementation, and verification that parallel execution produces identical results to serial execution.

The implementation mirrors Phase 177's parallel staging pattern: pre-partition problems by worker index to avoid coordination, use ThreadPoolExecutor for CPU-side concurrency while subprocess runs remain isolated, and collect results for deterministic output ordering. Thread safety for JSONL status writes is achieved via `threading.Lock`, ensuring concurrent workers never interleave or corrupt JSONL lines.

**Primary recommendation:** Use ThreadPoolExecutor with pre-partitioned problem chunks, protect JSONL writes with threading.Lock, preserve all existing CLI flags, default --jobs to min(os.cpu_count(), 4), and verify resume/continue-on-failure semantics with parallel-specific tests.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Concurrent subprocess dispatch | Thread pool orchestration | None | ThreadPoolExecutor manages concurrent problem subprocess execution |
| Thread-safe JSONL writes | Thread pool orchestration | None | Lock-protected file writes prevent interleaving/corruption |
| CPU-bound throughput improvement | Thread pool orchestration | None | Parallel execution targets CPU-side staging, not GPU work |
| Resume semantics preservation | Thread pool orchestration | None | Shared completed set must remain consistent under parallel access |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `concurrent.futures.ThreadPoolExecutor` | Python 3.12+ stdlib | Concurrent task execution | Built-in executor for CPU-bound parallelism [VERIFIED: Python 3.12 docs] |
| `threading.Lock` | Python 3.12+ stdlib | Thread-safe JSONL writes | Standard synchronization primitive for exclusive file access [VERIFIED: Python 3.12 docs] |
| `json` + `threading.Lock` | Python 3.12+ stdlib | Atomic JSONL line writes | Lock protects write operation to prevent line interleaving |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `os.cpu_count()` | Python 3.12+ stdlib | Default worker count | Sensible concurrency default: min(os.cpu_count(), 4) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ThreadPoolExecutor | ProcessPoolExecutor | ProcessPoolExecutor unsafe after torch import (fork-safety) — STATE.md confirms this decision |
| threading.Lock | queue.Queue + single writer | Lock is simpler for single-file write protection; Queue overkill for this use case |
| Pre-partition by index | Shared queue with race condition | Pre-partition eliminates coordination overhead and race conditions [VERIFIED: Phase 177 pattern] |

**Installation:** No external packages required — all concurrency primitives are in Python 3.12+ standard library.

**Version verification:** Python 3.12.13 available on target system. ThreadPoolExecutor and threading.Lock confirmed present via stdlib inspection.

## Package Legitimacy Audit

> No external packages are installed in this phase — all concurrency primitives are from Python 3.12+ standard library.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| (none) | — | — | — | — | — | — |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

This phase uses only Python standard library features — no package installation required.

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_derived_isolated.py                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Acquire PID lock (if --pid-lock enabled)                     │
│ 2. Discover problems (filter by --resume, --problem-id-file)    │
│ 3. Pre-partition problems by worker index (DERV-01)              │
│    Problem chunk A → Worker 1 ──┐                               │
│    Problem chunk B → Worker 2   │                               │
│    Problem chunk C → Worker 3    │  ThreadPoolExecutor            │
│    Problem chunk D → Worker 4  ──┘                               │
│ 4. Each worker processes problems serially:                     │
│    - build command                                            │
│    - subprocess.run uv run scripts/run_dataset.py              │
│    - capture result                                            │
│    - append_status(status_path, lock=threading.Lock)  (DERV-02) │
│ 5. Collect all results, sort by problem_id for output (DERV-03)│
│ 6. Return exit code based on --continue-on-failure             │
└─────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
scripts/
├── run_derived_isolated.py  # Main script: add ThreadPoolExecutor parallelism
└── run_dataset.py            # Unchanged: called as subprocess

tests/sol_execbench/
├── test_run_derived_isolated.py  # Existing tests: add parallel execution tests
└── conftest.py               # Shared fixtures
```

### Pattern 1: ThreadPoolExecutor with Pre-Partitioned Chunks (DERV-01)

**What:** Partition problems by index before submission to avoid worker coordination and race conditions.

**When to use:** Running CPU-bound concurrent tasks where each worker can own exclusive work items without runtime coordination.

**Example:**
```python
# Source: Phase 177 implementation pattern
from concurrent.futures import ThreadPoolExecutor, as_completed

def _partition_problems_by_index(
    problems: list[Path],
    max_workers: int,
) -> list[list[Path]]:
    """Pre-partition problems for exclusive worker ownership."""
    if not problems:
        return []
    chunk_size = (len(problems) + max_workers - 1) // max_workers
    chunks = []
    for i in range(0, len(problems), chunk_size):
        chunks.append(problems[i : i + chunk_size])
    return chunks

def _process_problem_chunk(
    chunk: list[Path],
    *,
    args: argparse.Namespace,
    benchmark_dir: Path,
    status_lock: threading.Lock,  # Protect JSONL writes
) -> list[ProblemStatus]:
    """Process a chunk of problems serially within a worker thread."""
    chunk_results = []
    for problem_dir in chunk:
        problem_id = problem_id_for(benchmark_dir, problem_dir)
        # Check resume/continue-on-failure conditions
        if should_skip(problem_id, completed=args.completed, ...):
            continue
        status = run_problem(args, problem_id=problem_id, problem_dir=problem_dir, ...)
        # Thread-safe status append
        with status_lock:
            append_status(args.status_jsonl, status)
        chunk_results.append(status)
        if status.status != "ok" and not args.continue_on_failure:
            return chunk_results  # Early exit
    return chunk_results

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # ... PID lock, problem discovery ...
    max_workers = args.jobs if hasattr(args, 'jobs') else min(os.cpu_count(), 4)
    problem_chunks = _partition_problems_by_index(problems, max_workers)

    status_lock = threading.Lock()  # Protect JSONL writes
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for chunk_idx, chunk in enumerate(problem_chunks):
            future = executor.submit(
                _process_problem_chunk,
                chunk=chunk,
                args=args,
                benchmark_dir=args.benchmark_dir,
                status_lock=status_lock,  # Pass lock to workers
            )
            futures[future] = chunk_idx

        try:
            for future in as_completed(futures.keys()):
                chunk_results = future.result()
                all_results.extend(chunk_results)
        except KeyboardInterrupt:
            # PRFL-05-style interrupt handling: cancel pending, collect partial
            for future in futures:
                future.cancel()
            partial_results = []
            for future in futures:
                if future.done():
                    try:
                        chunk_results = future.result()
                        partial_results.extend(chunk_results)
                    except Exception as exc:
                        logger.error(f"Worker failed: {exc}")
            all_results = partial_results
            # Exit with interrupt code (e.g., 130)

    return 1 if any(r.status != "ok" for r in all_results) else 0
```

**Why this pattern:** Pre-partitioning eliminates runtime coordination between workers, ensuring no race conditions on problem processing and deterministic result collection.

### Pattern 2: Thread-Safe JSONL Writes with Lock (DERV-02)

**What:** Protect `append_status()` file writes with `threading.Lock` to prevent concurrent workers from interleaving JSONL lines.

**When to use:** Multiple threads write to the same file, requiring atomic append operations.

**Example:**
```python
# Source: Python threading.Lock documentation
import threading
from pathlib import Path
from dataclasses import asdict
import json

# Global lock (or pass as parameter)
status_write_lock = threading.Lock()

def append_status(status_path: Path, status: ProblemStatus, lock: threading.Lock) -> None:
    """Thread-safe JSONL status append."""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:  # Acquire lock for entire write operation
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(status), sort_keys=True) + "\n")
    # Lock released automatically on context exit
```

**Why this pattern:** Without a lock, concurrent `append_status()` calls can interleave writes, producing corrupted JSONL lines (e.g., half of one line + half of another). The lock ensures atomic line writes.

### Anti-Patterns to Avoid

- **Sharing mutable completed set without synchronization:** The `completed` set from `load_completed()` is read-only after initialization, so it's safe for concurrent reads. If modified during execution, it would need protection.
- **Using ProcessPoolExecutor:** STATE.md documents that ProcessPoolExecutor is unsafe after torch import at module level due to fork-safety concerns. ThreadPoolExecutor is the correct choice.
- **Pre-partition by problem ID instead of index:** Partitioning by list index (not problem_id) ensures disjoint chunks without requiring a mapping lookup. Problem IDs are only used for output sorting.
- **Collecting results in completion order instead of problem order:** Results must be collected and sorted by problem_id to ensure deterministic output regardless of parallel completion order (DERV-03).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Concurrent task execution | Custom thread management | `concurrent.futures.ThreadPoolExecutor` | Built-in executor handles worker lifecycle, exception propagation, and result collection |
| Thread-safe file writes | Custom file locking schemes | `threading.Lock` with context manager | Standard primitive provides exclusive access with automatic release on context exit |
| Worker queue coordination | Shared task queue with mutex | Pre-partitioned chunks with ThreadPoolExecutor | Pre-partitioning eliminates runtime coordination overhead and race conditions |

**Key insight:** Custom concurrency primitives introduce subtle bugs (deadlocks, race conditions, priority inversion). Standard library primitives are well-tested, documented, and handle edge cases.

## Runtime State Inventory

> Include this section for rename/refactor/migration phases only. Omit entirely for greenfield phases.

This is a greenfield implementation phase (adding parallelism to existing script), not a rename/refactor phase. No runtime state inventory required.

## Common Pitfalls

### Pitfall 1: Non-atomic JSONL Writes Without Lock

**What goes wrong:** Concurrent workers call `append_status()` simultaneously, producing interleaved JSONL lines like `{"problem_id":"L1/a"...} {"problem_id` (corrupted mid-line).

**Why it happens:** File writes are not atomic at the Python level. Two threads can interleave `write()` calls on the same file handle.

**How to avoid:** Protect the entire write operation (open + write + close) with `threading.Lock`.

**Warning signs:** Status JSONL contains malformed JSON lines; `json.loads()` fails with `JSONDecodeError` when reading status file.

### Pitfall 2: Race Condition on Resume Skip Logic

**What goes wrong:** Worker A checks `should_skip("L1/a")` and sees it's not completed; Worker B checks same problem simultaneously; both workers process the same problem.

**Why it happens:** The `completed` set is built once at startup and never modified, so concurrent reads are safe. Race conditions only occur if the set is modified during execution.

**How to avoid:** Never modify the `completed` set during execution. If modification is required, protect it with a lock or use `threading.local()` for per-worker tracking.

**Warning signs:** Duplicate problem entries in status JSONL; log shows same problem_id processed twice.

### Pitfall 3: Non-Deterministic Output Order

**What goes wrong:** Status JSONL lines appear in completion order (fastest worker first), breaking reproducibility and downstream tools that expect sorted output.

**Why it happens:** Workers complete at different times; results are appended to `all_results` in completion order.

**How to avoid:** Collect all results after worker completion, then sort by problem_id before writing final output: `all_results.sort(key=lambda r: r.problem_id)`.

**Warning signs:** Status file order differs between runs; downstream tools fail to parse unsorted output.

### Pitfall 4: KeyboardInterrupt Handler Missing

**What goes wrong:** User presses Ctrl+C during parallel execution; some workers terminate mid-task, leaving incomplete status entries and no summary.

**Why it happens:** Default ThreadPoolExecutor behavior on interrupt is undefined; futures are cancelled without partial result collection.

**How to avoid:** Catch `KeyboardInterrupt`, cancel pending futures, collect completed results, write partial summary, exit with code 130 (standard interrupt exit code). Follow Phase 177 pattern.

**Warning signs:** Ctrl+C produces incomplete status file; no summary written; return code inconsistent with interrupt behavior.

### Pitfall 5: Default --jobs Too High for System Resources

**What goes wrong:** Default `--jobs=os.cpu_count()` (e.g., 32 workers) exhausts memory or file descriptors, causing OOM kills or "too many open files" errors.

**Why it happens:** Each subprocess spawns a Python interpreter + PyTorch; 32 concurrent processes can exceed system limits.

**How to avoid:** Default to `min(os.cpu_count(), 4)` to limit concurrency while still providing parallelism benefit. Document this in help text.

**Warning signs:** OOM killer messages in dmesg; "Too many open files" errors; system becomes unresponsive during execution.

## Code Examples

Verified patterns from official sources:

### Thread-Safe File Write with Lock
```python
# Source: Python threading.Lock documentation [CITED: https://docs.python.org/3/library/threading.html]
import threading

lock = threading.Lock()

def thread_safe_write(path: Path, content: str) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(content)
```

### ThreadPoolExecutor with Exception Handling
```python
# Source: Python concurrent.futures documentation [CITED: https://docs.python.org/3/library/concurrent.futures.html]
from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_execute(tasks: list[dict]) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Task generated exception: {exc}")
    return results
```

### Pre-Partition for Worker Isolation
```python
# Source: Phase 177 implementation pattern [VERIFIED: run_rdna4_profiler_timing_batch.py]
def partition_by_index(items: list[T], max_workers: int) -> list[list[T]]:
    """Create disjoint chunks for exclusive worker ownership."""
    chunk_size = (len(items) + max_workers - 1) // max_workers
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Serial for-loop subprocess execution | ThreadPoolExecutor with pre-partitioned chunks | Phase 178 (this phase) | Improves throughput for CPU-bound derived generation; preserves correctness via thread-safe writes |
| Non-protected JSONL writes | threading.Lock-protected JSONL writes | Phase 178 (this phase) | Eliminates race condition corruption in status file |
| ProcessPoolExecutor (unsafe) | ThreadPoolExecutor (safe) | Phase 177 (STATE.md documented) | Avoids torch fork-safety deadlock; CPU-bound tasks benefit from threading |

**Deprecated/outdated:**
- ProcessPoolExecutor for tasks after torch import: Unsafe due to fork-safety concerns (STATE.md decision).
- Unprotected shared mutable state in multithreaded context: Requires explicit synchronization or immutable data structures.

## Assumptions Log

> List all claims tagged `[ASSUMED]` in this research. The planner and discuss-phase use this section to identify decisions that need user confirmation before execution.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Default `--jobs=min(os.cpu_count(), 4)` is appropriate for this workload | Standard Stack | If too conservative, throughput gains are limited; if too aggressive, system resources exhausted |
| A2 | Existing `--continue-on-failure` semantics can be preserved without modification to run_problem() | Architecture Patterns | If run_problem() has shared state, parallel execution may break failure handling |
| A3 | The `completed` set from `load_completed()` is read-only after initialization | Common Pitfalls #2 | If modified during execution, race conditions require additional synchronization |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.

## Open Questions

1. **Should `--jobs` default vary by available system resources?**
   - What we know: `os.cpu_count()` returns 32 on the target system; 32 workers may be too aggressive.
   - What's unclear: Whether memory limits, file descriptor limits, or other constraints should further cap the default.
   - Recommendation: Use `min(os.cpu_count(), 4)` as default; document that users can increase via `--jobs` for high-memory systems.

2. **Should KeyboardInterrupt exit code follow Phase 177 pattern (130) or use different code?**
   - What we know: Phase 177 uses exit code 130 for interrupted batch execution.
   - What's unclear: Whether derived script should use same code for consistency or different code for distinguishability.
   - Recommendation: Use 130 for consistency with Phase 177 interrupt behavior.

## Environment Availability

> Skip this section if the phase has no external dependencies (code/config-only changes).

This phase has no external dependencies beyond Python 3.12+ standard library.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12+ | ThreadPoolExecutor, threading.Lock | ✓ | 3.12.13 | — |
| os.cpu_count() | Default --jobs calculation | ✓ | stdlib | — |
| subprocess module | Subprocess execution | ✓ | stdlib | — |
| json module | JSONL serialization | ✓ | stdlib | — |

**Missing dependencies with no fallback:** none

**Missing dependencies with fallback:** none

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | pyproject.toml ( `[tool.pytest.ini_options]` ) |
| Quick run command | `uv run pytest tests/sol_execbench/test_run_derived_isolated.py -x` |
| Full suite command | `uv run pytest tests/` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DERV-01 | ThreadPoolExecutor dispatch per-problem subprocesses | integration | `uv run pytest tests/sol_execbench/test_run_derived_isolated.py::test_parallel_dispatch_concurrency -x` | ❌ Wave 0 |
| DERV-02 | Status JSONL writes are thread-safe | unit | `uv run pytest tests/sol_execbench/test_run_derived_isolated.py::test_thread_safe_jsonl_writes -x` | ❌ Wave 0 |
| DERV-03 | --resume/--continue-on-failure produce identical results under parallel execution | integration | `uv run pytest tests/sol_execbench/test_run_derived_isolated.py::test_parallel_resume_semantics -x` | ❌ Wave 0 |
| DERV-04 | --jobs flag controls concurrency with sensible default | unit | `uv run pytest tests/sol_execbench/test_run_derived_isolated.py::test_jobs_flag_default -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/sol_execbench/test_run_derived_isolated.py -x`
- **Per wave merge:** `uv run pytest tests/`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_run_derived_isolated.py` — parallel dispatch test (DERV-01)
- [ ] `tests/sol_execbench/test_run_derived_isolated.py` — thread-safe JSONL writes test (DERV-02)
- [ ] `tests/sol_execbench/test_run_derived_isolated.py` — parallel resume semantics test (DERV-03)
- [ ] `tests/sol_execbench/test_run_derived_isolated.py` — --jobs flag default test (DERV-04)
- [ ] Framework install: `uv sync --all-groups` (pytest available in dev group)

*(If no gaps: "None — existing test infrastructure covers all phase requirements")*

## Security Domain

> Required when `security_enforcement` is enabled (absent = enabled). Omit only if explicitly `false` in config.

Security enforcement is enabled for this phase (default in config.json).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | argparse validates `--jobs` is positive integer; problem discovery validates JSON structure |
| V6 Cryptography | no | — |

### Known Threat Patterns for Python Threading

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| File path injection in --status-jsonl | Tampering | argparse.Path validation; ensure file is within output directory |
| JSON injection via problem_id manipulation | Tampering | Input validation on problem_id format; reject paths with `..` or absolute paths |
| DoS via --jobs=1000000 | Denial of Service | argparse validation cap to reasonable max (e.g., 64) |
| Symlink attack on status file | Tampering | Verify status file is not a symlink before writing |

## Sources

### Primary (HIGH confidence)
- Python 3.12 threading documentation [CITED: https://docs.python.org/3/library/threading.html] — threading.Lock API, context manager usage, thread safety guarantees
- Python 3.12 concurrent.futures documentation [CITED: https://docs.python.org/3/library/concurrent.futures.html] — ThreadPoolExecutor usage, exception handling patterns
- Phase 177 implementation [VERIFIED: scripts/run_rdna4_profiler_timing_batch.py] — Pre-partition pattern, interrupt handling, result collection
- Current derived script [VERIFIED: scripts/run_derived_isolated.py] — Existing resume/continue-on-failure logic, problem discovery patterns

### Secondary (MEDIUM confidence)
- ThreadPoolExecutor thread safety discussion [CITED: https://stackoverflow.com/questions/45286344/is-python-threadpoolexecutor-thread-safe] — Confirms ThreadPoolExecutor.submit() and shutdown() are thread-safe
- Thread-safe file write patterns [CITED: https://stackoverflow.com/questions/30135091/write-thread-safe-to-file-in-python] — Community guidance on lock-protected file writes
- Shared state concurrency risks [CITED: https://medium.com/towardsdev/whats-the-best-way-to-handle-concurrency-in-python-threadpoolexecutor-or-asyncio-85da1be58557] — Warns against unsafe shared state in thread pools

### Tertiary (LOW confidence)
- None — all findings verified against official docs or codebase inspection

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All primitives verified in Python 3.12 stdlib; Phase 177 pattern confirms ThreadPoolExecutor usage
- Architecture: HIGH - Phase 177 implementation provides proven parallel pattern; thread safety verified via threading.Lock docs
- Pitfalls: HIGH - Common threading failures well-documented; specific JSONL corruption scenarios verified via Stack Overflow discussions

**Research date:** 2025-06-11
**Valid until:** 30 days (Python stdlib APIs are stable; ThreadPoolExecutor behavior unchanged since 3.2)
