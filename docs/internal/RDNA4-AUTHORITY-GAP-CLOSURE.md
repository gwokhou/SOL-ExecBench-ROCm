# RDNA4 Authority Gap Closure

This document closes two v1.33 audit gaps as explicit release boundaries:

- incomplete full profiler-backed timing coverage;
- missing benchmark-grade timing authority.

These gaps are closed for release/audit purposes as **deferred blockers**, not
as achieved benchmark evidence.

## Full Profiler-Backed Timing Coverage

Current accepted coverage:

- denominator: 235 problems;
- profiler-backed: 61 problems;
- fallback timing: 45 problems;
- current-device OOM blocked: 13 problems;
- profiler blocked: 2 problems;
- readiness blocked: 114 problems.

Closure status: `closed_as_deferred_blocker`

The release bundle must not claim full profiler-backed timing coverage. The
accepted denominator ledger makes the shortfall explicit and preserves
`full_profiler_backed_timing_coverage=false`.

Future work may reopen this gap only with new complete profiler-backed timing
sidecars or a larger/different RDNA4 host that resolves current-device memory
blockers without changing benchmark semantics.

## Benchmark-Grade Timing Authority

Phase 167 verified:

- `rocm-smi` clock-control command path;
- manual SCLK/MCLK set command path;
- reset path returning the device to `Performance Level: auto`.

Phase 167 did not verify:

- stable observed maximum SCLK/MCLK during a benchmark window.

Closure status: `closed_as_deferred_blocker`

The release bundle must not claim benchmark-grade authoritative timing.
Profiler-backed timing may be cited for the scoped subset, and fallback timing
may remain diagnostic, but authoritative timing remains unsupported until a
future run records stable benchmark-window clock lock and reset evidence.

## Claim Rule

Safe statement:

> RDNA4 `gfx1200` has release evidence with explicit profiler coverage and
> timing authority blockers. Full profiler-backed timing coverage and
> benchmark-grade timing authority are closed as deferred blockers, not
> achieved claims.

Forbidden statements:

- "RDNA4 has complete profiler-backed timing coverage."
- "RDNA4 timing is benchmark-grade."
- "The v1.33 release is leaderboard-ready."
- "The timing evidence proves paper parity."
