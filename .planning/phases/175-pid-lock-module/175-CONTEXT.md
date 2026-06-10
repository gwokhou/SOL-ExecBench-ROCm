# Phase 175: PID Lock Module - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — no grey areas)

<domain>
## Phase Boundary

Scripts that must run exclusively can detect and reject concurrent instances
with a kernel-managed lock that never leaves stale state. Deliver a reusable
`fcntl.flock`-based context manager in `src/sol_execbench/core/bench/pid_lock.py`
that scripts acquire at startup. The lock must auto-release on process death
(SIGKILL, OOM) with no stale-lock cleanup required.

Requirements: INST-01, INST-02, INST-03

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure
phase. Key constraints from research and requirements:

- Use stdlib `fcntl.flock(LOCK_EX | LOCK_NB)` — not PID-in-file, not filelock
- Lock file location: `{output_dir}/.sol-execbench.lock` (co-located with output)
- Context manager API: `acquire_pid_lock(output_dir) -> ProcessLock`
- On contention: exit with diagnostic showing lock file path, suggest waiting
  or removing stale lock if confident no other instance holds it
- Mandatory for `run_rdna4_profiler_timing_batch.py` and
  `run_rdna4_profiler_overhead_calibration.py`; optional `--pid-lock` flag for
  `run_derived_isolated.py`
- Module location: `src/sol_execbench/core/bench/pid_lock.py`
- Unit tests must verify: exclusive acquire, contention rejection, auto-release
  on process exit (subprocess-based test)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/run_dataset.py` — already uses `concurrent.futures.ThreadPoolExecutor`
  as reference pattern for future phases
- `src/sol_execbench/core/bench/` — existing bench utilities (timing, clock_lock,
  rocm_profiler) establish the module placement convention

### Established Patterns
- Utility modules in `core/bench/` use `@dataclass` or plain functions
- Scripts import from `sol_execbench.core.bench` — new module follows same path
- Error exits use `sys.exit(1)` with `print()` to stderr in scripts

### Integration Points
- `scripts/run_rdna4_profiler_timing_batch.py` — add `acquire_pid_lock()` call
  near `main()` entry, before any file I/O
- `scripts/run_rdna4_profiler_overhead_calibration.py` — same pattern
- `scripts/run_derived_isolated.py` — add `--pid-lock` argparse flag, conditional
  acquire

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. Follow ROADMAP success criteria
and codebase conventions.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.
