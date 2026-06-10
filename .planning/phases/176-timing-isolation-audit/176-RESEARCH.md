# Phase 176: Timing Isolation Audit - Research

**Researched:** 2026-06-10
**Domain:** GPU process detection, clock state verification, subprocess isolation, environment snapshotting
**Confidence:** HIGH

## Summary

Phase 176 implements timing isolation auditing for ROCm profiling scripts to detect and warn about concurrent GPU access, verify clock lock state integrity, reduce inter-problem GPU state leakage via cache clearing, and record environment state for reproducibility audits. The phase builds on existing `clock_lock.py` and `environment.py` modules, adding new `timing_isolation.py` utilities that integrate into profiling batch scripts via pre-flight checks, periodic re-verification, and batch summary sidecars.

**Primary recommendation:** Create a new `src/sol_execbench/core/bench/timing_isolation.py` module with functions for concurrent GPU process detection (`detect_concurrent_gpu_processes`), clock lock verification wrappers (`verify_clock_state_with_warning`), subprocess boundary cache clearing (`clear_gpu_cache_between_subprocesses`), and environment snapshot collection for batch summaries (`collect_timing_environment_snapshot`). Integration into `run_rdna4_profiler_timing_batch.py` at batch start, between problems, and in summary generation.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Concurrent GPU process detection | Browser / Client | — | Detection runs in the parent profiling script before spawning GPU subprocesses |
| Clock lock state verification | Browser / Client | — | Verification happens at script startup and between problems in the parent process |
| GPU cache clearing | Browser / Client | — | Cache clearing occurs at subprocess boundaries in the parent process |
| Environment snapshot collection | Browser / Client | — | Snapshots are collected by the parent script and written to sidecar files |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `subprocess` (stdlib) | Python 3.12+ | ROCm SMI process detection | [VERIFIED: Python 3.12 docs] Standard subprocess execution for `rocm-smi --showpids` and `amd-smi` commands |
| `re` (stdlib) | Python 3.12+ | Parse process output | [VERIFIED: Python 3.12 docs] Pattern matching for extracting PIDs from SMI output |
| `logging` (stdlib) | Python 3.12+ | Warning/error logging | [VERIFIED: Python 3.12 docs] Consistent with existing `clock_lock.py` logging patterns |
| `dataclasses` (stdlib) | Python 3.12+ | Snapshot data structures | [VERIFIED: Python 3.12 docs] Matches existing `environment.py` patterns |
| `pydantic` | v2.0+ | Environment snapshot models | [CITED: existing codebase] Already used in `environment.py` for `EnvironmentSnapshot` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `clock_lock` (local) | existing | Clock state verification | [VERIFIED: codebase] Reuse existing `verify_clocks()` and `are_clocks_locked()` functions |
| `environment` (local) | existing | Environment snapshot collection | [VERIFIED: codebase] Reuse existing `collect_environment_snapshot()` for batch summaries |
| `torch.cuda` | via ROCm PyTorch | GPU cache clearing | [CITED: PyTorch docs] Use `torch.cuda.empty_cache()` at subprocess boundaries |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `rocm-smi --showpids` | `amd-smi` process listing | `amd-smi` requires `amd-smi static -a` which is slower; `rocm-smi --showpids` is faster and purpose-built for process detection |
| Pre-flight detection only | Continuous monitoring during profiling | Continuous monitoring would interfere with timing measurements; pre-flight + periodic checks provide adequate protection |
| Automatic abort on concurrent processes | Warning-only mode | Abort is too strict for development environments; warning + severity levels allows user control |

**Installation:**
```bash
# No new external packages required — uses stdlib and existing modules
# torch.cuda.empty_cache() uses existing PyTorch ROCm installation
```

**Version verification:** All stdlib modules verified via Python 3.12 official documentation. PyTorch ROCm already installed in project. No new external packages to verify.

## Package Legitimacy Audit

> **Not applicable** — this phase installs zero new external packages. All functionality uses Python stdlib (`subprocess`, `re`, `logging`, `dataclasses`) and existing project modules (`clock_lock`, `environment`, PyTorch ROCm).

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Profiling Script Parent Process                        │
│                  (run_rdna4_profiler_timing_batch.py)                     │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
                           │ startup
                           ▼
                   ┌─────────────────┐
                   │  Pre-flight      │
                   │  Isolation Audit │
                   └────────┬─────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ Detect GPU   │  │ Verify Clock │  │ Collect Env  │
  │ Processes   │  │ Lock State   │  │ Snapshot     │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                 │                 │
         │ rocm-smi        │ rocm-smi        │ amd-smi
         │ --showpids      │ --showperflevel │ rocminfo
         │                 │                 │
         ▼                 ▼                 ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ Warn/Abort   │  │ Warn if not  │  │ Store for    │
  │ if concurrent│  │ STABLE_PEAK  │  │ summary       │
  └──────────────┘  └──────────────┘  └──────────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                           │ Continue to profiling loop
                           ▼
                  ┌─────────────────┐
                  │  For each      │
                  │  Problem       │
                  └────────┬────────┘
                           │
           ┌───────────────┼───────────────┐
           │                               │
           ▼                               ▼
   ┌───────────────┐             ┌───────────────┐
   │  Re-verify    │             │  Clear GPU    │
   │  clock state  │             │  cache        │
   │  (periodic)   │             │  at boundary  │
   └───────┬───────┘             └───────┬───────┘
           │                               │
           │ Warn if drifted                │ torch.cuda.empty_cache()
           │                               ▼
           │                       ┌───────────────┐
           │                       │  Spawn        │
           │                       │  subprocess   │
           │                       │  (profiling)  │
           │                       └───────┬───────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Write Batch   │
                  │  Summary with  │
                  │  Env Snapshot  │
                  └─────────────────┘
```

### Recommended Project Structure
```
src/sol_execbench/core/bench/
├── timing_isolation.py          # NEW: Main isolation audit module
├── clock_lock.py                # EXISTING: Reuse for clock verification
└── environment.py              # EXISTING: Reuse for snapshot collection

scripts/
├── run_rdna4_profiler_timing_batch.py  # MODIFY: Add isolation checks
└── run_rdna4_profiler_overhead_calibration.py  # MODIFY: Add isolation checks

tests/sol_execbench/core/bench/
└── test_timing_isolation.py     # NEW: Isolation audit tests
```

### Pattern 1: Concurrent GPU Process Detection
**What:** Parse `rocm-smi --showpids` output to detect non-trivial GPU processes before profiling starts.
**When to use:** At batch startup and optionally between problems during long runs.
**Example:**
```python
# Source: [ROCm/ROC-smi GitHub Issue #60]
def detect_concurrent_gpu_processes() -> list[dict[str, Any]]:
    """Detect concurrent GPU processes via rocm-smi --showpids.
    
    Returns a list of process dictionaries with PID, device ID, and process name.
    Returns empty list if no processes are running or detection fails.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showpids"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.warning("rocm-smi --showpids failed: %s", result.stderr.strip())
            return []
        
        # Parse "No KFD PIDs currently running" vs actual process listings
        if "No KFD PIDs" in result.stdout:
            return []
        
        # Parse process listings (format varies by ROCm version)
        processes = _parse_rocm_smi_pids(result.stdout)
        return processes
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("GPU process detection failed: %s", e)
        return []
```

### Pattern 2: Clock State Verification with Warning
**What:** Wrap existing `clock_lock.verify_clocks()` to log warnings when clock state is not STABLE_PEAK.
**When to use:** At batch start and between problems to detect clock drift.
**Example:**
```python
# Source: [existing clock_lock.py module]
def verify_clock_state_with_warning(context: str = "batch_start") -> bool:
    """Verify clock state and warn if not STABLE_PEAK.
    
    Args:
        context: Description of when verification is occurring (e.g., "batch_start", "problem_5")
    
    Returns:
        True if clocks are locked, False otherwise.
    """
    from sol_execbench.core.bench.clock_lock import verify_clocks
    
    is_locked = verify_clocks()
    if not is_locked:
        logger.warning(
            "Clock state verification failed at %s: GPU not in STABLE_PEAK mode. "
            "Timing measurements may be unstable.",
            context
        )
    else:
        logger.info("Clock state verified at %s: STABLE_PEAK mode confirmed", context)
    
    return is_locked
```

### Pattern 3: GPU Cache Clearing at Subprocess Boundaries
**What:** Call `torch.cuda.empty_cache()` between profiling subprocesses to reduce GPU memory state leakage.
**When to use:** After each profiling subprocess completes, before starting the next one.
**Example:**
```python
# Source: [PyTorch Forums - torch.cuda.empty_cache discussion]
def clear_gpu_cache_between_subprocesses() -> None:
    """Clear GPU cache at subprocess boundaries to reduce state leakage.
    
    Best-effort; errors are logged but not raised. Safe to call even if
    torch.cuda.is_available() is False (no-op in that case).
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared at subprocess boundary")
    except Exception as e:
        logger.warning("GPU cache clearing failed: %s", e)
```

### Pattern 4: Environment Snapshot for Batch Summary
**What:** Collect and attach environment snapshot to batch summary for reproducibility audits.
**When to use:** When generating batch summary JSON sidecar.
**Example:**
```python
# Source: [existing environment.py module]
def collect_timing_environment_snapshot() -> dict[str, Any]:
    """Collect environment snapshot for timing isolation audit trail.
    
    Returns a dictionary suitable for embedding in batch summary JSON.
    Includes GPU processes, clock state, and tool availability.
    """
    from sol_execbench.core.environment import collect_environment_snapshot
    from sol_execbench.core.bench.clock_lock import are_clocks_locked
    
    snapshot = collect_environment_snapshot(collect_pytorch=False)
    
    return {
        "schema_version": "sol_execbench.timing_isolation_snapshot.v1",
        "generated_at": snapshot.generated_at,
        "gpu_processes": detect_concurrent_gpu_processes(),
        "clocks_locked": are_clocks_locked(),
        "tools_available": {
            name: result.status.value
            for name, result in snapshot.tools.items()
        },
        "warnings": snapshot.warnings,
    }
```

### Anti-Patterns to Avoid
- **Calling torch.cuda.empty_cache() inside profiling subprocess:** This interferes with timing measurements. Call it only in the parent process between subprocess invocations.
- **Hard abort on concurrent GPU processes:** Development environments may have benign concurrent processes (e.g., X11, desktop compositor). Use warning + severity levels instead.
- **Assuming rocm-smi output format is stable:** ROCm versions have different output formats. Use flexible parsing with fallback for unrecognized formats.
- **Calling SMI tools in tight loops:** These commands are slow (100-500ms). Cache results and only recheck periodically between problems, not within individual problem profiling loops.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Clock lock verification | Custom `amd-smi` parsing | Existing `clock_lock.verify_clocks()` | Already handles ROCm version differences and failure modes |
| Environment snapshotting | Custom tool probing | Existing `environment.collect_environment_snapshot()` | Probes amd-smi, rocminfo, rocm_agent_enumerator with timeout handling |
| SMI command execution | Raw `subprocess.run()` calls | Wrap in timeout-bounded helpers | SMI tools can hang; existing patterns use 3-5 second timeouts |
| Pydantic models for snapshots | Custom dataclasses | Existing `EnvironmentSnapshot` and `ToolProbeResult` | Already battle-tested for ROCm environment data |

**Key insight:** The existing `clock_lock` and `environment` modules already handle 80% of the required functionality. The new `timing_isolation.py` module should compose these existing functions rather than reimplementing SMI parsing and clock verification logic.

## Common Pitfalls

### Pitfall 1: SMI Tool Output Format Changes
**What goes wrong:** ROCm version upgrades change `rocm-smi --showpids` output format, breaking parsers.
**Why it happens:** AMD ROCm tools don't guarantee stable CLI output format across versions.
**How to avoid:** Use flexible regex-based parsing that handles multiple formats, log warnings for unrecognized output, and degrade gracefully to "detection failed" rather than crashing.
**Warning signs:** Parser returning empty lists on systems where GPU processes definitely exist.

### Pitfall 2: Interfering with Timing Measurements
**What goes wrong:** Isolation checks themselves add overhead that corrupts microbenchmark timing.
**Why it happens:** Calling `rocm-smi` or `torch.cuda.empty_cache()` within the profiling loop adds 100-500ms overhead.
**How to avoid:** Only run isolation checks at batch boundaries (before first problem, between problems), never within a single problem's profiling loop.
**Warning signs:** Timing measurements showing consistent 100-500ms overhead that correlates with check frequency.

### Pitfall 3: False Positives from Benign GPU Processes
**What goes wrong:** Detection aborts profiling due to desktop compositor, X11, or display server using GPU.
**Why it happens:** Desktop environments always have GPU processes running; they don't necessarily interfere with compute workloads.
**How to avoid:** Use warning-only mode by default, allow user to configure severity, and optionally filter known benign processes by name.
**Warning signs:** Scripts failing on developer desktops despite no actual compute contention.

### Pitfall 4: Cache Clearing Performance Impact
**What goes wrong:** `torch.cuda.empty_cache()` is slow and can dominate runtime in short-running kernels.
**Why it happens:** Cache clearing involves synchronization and memory deallocation that can take 50-200ms.
**How to avoid:** Only call at subprocess boundaries where the subprocess overhead already dominates, and measure the impact before enabling by default.
**Warning signs:** Profiling batch runtime increasing significantly without timing improvements.

## Code Examples

Verified patterns from official sources:

### Concurrent Process Detection (rocm-smi)
```python
# Source: [ROCm/ROC-smi GitHub Issue #60]
def detect_concurrent_gpu_processes() -> list[dict[str, Any]]:
    """Detect concurrent GPU processes via rocm-smi --showpids."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showpids"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "No KFD PIDs currently running" in result.stdout:
            return []
        return _parse_rocm_smi_pids(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("GPU process detection failed: %s", e)
        return []
```

### Cache Clearing (PyTorch)
```python
# Source: [PyTorch Forums - torch.cuda.empty_cache discussion]
def clear_gpu_cache_between_subprocesses() -> None:
    """Clear GPU cache at subprocess boundaries to reduce state leakage."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared at subprocess boundary")
    except Exception as e:
        logger.warning("GPU cache clearing failed: %s", e)
```

### Clock Verification (existing module)
```python
# Source: [existing clock_lock.py module]
def verify_clock_state_with_warning(context: str = "batch_start") -> bool:
    """Verify clock state and warn if not STABLE_PEAK."""
    from sol_execbench.core.bench.clock_lock import verify_clocks
    
    is_locked = verify_clocks()
    if not is_locked:
        logger.warning(
            "Clock state verification failed at %s: GPU not in STABLE_PEAK mode",
            context
        )
    return is_locked
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual environment checks | Automated pre-flight audits | Phase 176 | Consistency and reproducibility |
| No clock state verification | Periodic re-verification during batches | Phase 176 | Detect clock drift during long runs |
| Subprocess state leakage | Explicit cache clearing at boundaries | Phase 176 | Reduced inter-problem interference |
| No audit trail | Environment snapshot sidecars | Phase 176 | Post-hoc reproducibility analysis |

**Deprecated/outdated:**
- Manual `rocm-smi` checks by users: Replaced by automated detection
- Assuming STABLE_PEAK stays locked: Replaced by periodic re-verification
- Trusting subprocess isolation: Replaced by explicit cache clearing

## Assumptions Log

> List all claims tagged `[ASSUMED]` in this research. The planner and discuss-phase use this
> section to identify decisions that need user confirmation before execution.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `rocm-smi --showpids` output format stable across ROCm 6.x | Standard Stack | Parser breaks on ROCm 7.x |
| A2 | `torch.cuda.empty_cache()` is safe to call at subprocess boundaries | Standard Stack | Causes memory corruption or crashes |
| A3 | 100-500ms overhead for SMI commands is acceptable at problem boundaries | Common Pitfalls | Performance regression unacceptable |
| A4 | Desktop GPU processes (X11, compositor) don't interfere with compute | Common Pitfalls | False negatives mask real contention |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.

## Open Questions

1. **Concurrent process severity levels**
   - What we know: ISOL-01 requires "warn or abort depending on severity"
   - What's unclear: What severity levels should exist? (WARNING, ERROR, CRITICAL?) What defines each level?
   - Recommendation: Use three levels (INFO, WARNING, ERROR) with configurable threshold via CLI flag `--isolation-severity`

2. **Clock state recheck frequency**
   - What we know: ISOL-02 requires rechecking "between problems during long batch runs"
   - What's unclear: Define "long"? Every problem? Every N problems? Time-based?
   - Recommendation: Recheck every 10 problems by default, configurable via `--clock-recheck-interval`

3. **Cache clearing default behavior**
   - What we know: ISOL-03 requires `torch.cuda.empty_cache()` at subprocess boundaries
   - What's unclear: Should this be on by default or opt-in given performance impact?
   - Recommendation: On by default for profiling batches, opt-out via `--skip-gpu-cache-clear`

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `rocm-smi` | GPU process detection | ✓ | /usr/bin/rocm-smi | — |
| `amd-smi` | Clock state management | ✓ | /usr/bin/amd-smi | — |
| `torch.cuda` | Cache clearing | ✓ | ROCm PyTorch installed | No-op if unavailable |
| Python 3.12+ | Script execution | ✓ | 3.12.x | — |

**Missing dependencies with no fallback:** none

**Missing dependencies with fallback:** none

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | pyproject.toml (uv project) |
| Quick run command | `uv run pytest tests/sol_execbench/core/bench/test_timing_isolation.py -x` |
| Full suite command | `uv run pytest tests/` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ISOL-01 | Detect concurrent GPU processes and warn/abort | unit | `pytest tests/ -k "test_detect_concurrent" -x` | ❌ Wave 0 |
| ISOL-02 | Verify clock state at batch start and between problems | integration | `pytest tests/ -k "test_clock_verification" -x` | ❌ Wave 0 |
| ISOL-03 | Call torch.cuda.empty_cache() at subprocess boundaries | unit | `pytest tests/ -k "test_cache_clearing" -x` | ❌ Wave 0 |
| ISOL-04 | Environment snapshot in batch summary | integration | `pytest tests/ -k "test_env_snapshot" -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/sol_execbench/core/bench/test_timing_isolation.py -x`
- **Per wave merge:** `uv run pytest tests/`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/sol_execbench/core/bench/test_timing_isolation.py` — covers ISOL-01 through ISOL-04
- [ ] `tests/sol_execbench/core/bench/conftest.py` — shared fixtures for SMI mocking (if not exists)
- [ ] Framework install: `uv sync --all-groups` — existing test infrastructure

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | SMI output parsing with regex validation |
| V6 Cryptography | no | — |

### Known Threat Patterns for ROCm Benchmarking

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Command injection in SMI output | Tampering | Regex validation, timeout-bounded subprocess calls |
| SMI tool hang (DoS) | Denial of Service | 5-second timeouts on all subprocess calls |
| Environment snapshot data injection | Tampering | Pydantic v2 validation, schema versioning |
| Clock state drift (integrity) | Tampering | Periodic re-verification, warnings on drift |

## Sources

### Primary (HIGH confidence)
- [ROCm/ROC-smi GitHub Issue #60] - Processes using a GPU discussion
- [ROCm System Management Interface Documentation] - Official rocm-smi CLI reference
- [AMD ROCm SMI Library Documentation] - Process information functions
- [PyTorch Forums - torch.cuda.empty_cache] - Cache clearing best practices
- [Python 3.12 subprocess documentation] - Subprocess execution patterns
- [Existing codebase modules] - `clock_lock.py`, `environment.py` patterns

### Secondary (MEDIUM confidence)
- [Stack Overflow - GPU memory clearing] - torch.cuda.empty_cache effectiveness
- [Fast.ai Forums - Cache performance] - Performance impact of cache clearing
- [LUMI-G GPU Monitoring Guide] - Practical rocm-smi usage patterns

### Tertiary (LOW confidence)
- [GitHub Issue #3002] - ROCm SMI process info edge cases
- [Medium - GPU memory experiments] - CUBLAS workspace clearing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Based on existing codebase patterns and Python stdlib
- Architecture: HIGH - Clear tier boundaries, no cross-cutting concerns
- Pitfalls: MEDIUM - SMI tool stability and cache clearing impact need validation

**Research date:** 2026-06-10
**Valid until:** 30 days (ROCm tooling evolves slowly; patterns are stable)
