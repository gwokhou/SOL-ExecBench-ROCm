# RDNA4 Authority Gap Closure

This document closes RDNA4 audit gaps as explicit release boundaries:

- incomplete full profiler-backed timing coverage;
- missing benchmark-grade timing authority.

These gaps are closed for release/audit purposes as **deferred blockers**, not
as achieved benchmark evidence.

## Full Profiler-Backed Timing Coverage

The accepted historical v1.35 rerun coverage was generated under the now-pruned
ignored local output tree `out/rdna4-v135-rerun-20260611/`; its recorded totals are:

- denominator: 235 problems;
- full profiler-backed: 88 problems;
- partial profiler-backed: 28 problems;
- fallback timing: 0 problems;
- profiler blocked: 0 problems;
- ready but missing profiler timing: 73 problems;
- reference/current-device OOM blocked: 5 problems;
- readiness blocked: 41 problems.

The v1.35 profiler timing batch produced 121 replacement timing sidecars and
121 workload manifests, with no remaining resume targets. Its batch summary
records `failed=0`, `fallback_or_missing=0`, `interrupted=false`, and
`profiler_blocked=0`.

Closure status: `closed_as_deferred_blocker`

The release bundle must not claim full profiler-backed timing coverage. The
accepted denominator ledger makes the shortfall explicit and preserves
`full_profiler_backed_timing_coverage=false`.

Future work may reopen this gap only with new complete profiler-backed timing
sidecars or a larger/different RDNA4 host that resolves current-device memory
blockers without changing benchmark semantics.

## Benchmark-Grade Timing Authority

Prior RDNA4 evidence verified:

- `rocm-smi` clock-control command path;
- manual SCLK/MCLK set command path;
- reset path returning the device to `Performance Level: auto`.

The v1.35 rerun adds stricter execution evidence:

- strict-isolation profiler timing runs on one GPU device;
- `rocprofv3` overhead calibration evidence;
- clock-lock/audit sidecars for the measured rerun;
- no cross-report consistency findings in the rebuilt same-source reports.

Closure status: `closed_as_deferred_blocker`

The release bundle still must not claim benchmark-grade authoritative timing.
Profiler-backed timing may be cited for the scoped subset, but authoritative
timing remains unsupported until full profiler-backed coverage and a release
policy for benchmark-grade timing authority are both satisfied.

## Claim Rule

Safe statement:

> RDNA4 `gfx1200` has release evidence with explicit profiler coverage and
> timing authority blockers. The v1.35 rerun records 88 full profiler-backed
> problems and 28 partial profiler-backed problems out of the 235-problem
> denominator, with 0 consistency findings in the rebuilt reports. Full
> profiler-backed timing coverage and benchmark-grade timing authority remain
> deferred blockers, not achieved claims.

Forbidden statements:

- "RDNA4 has complete profiler-backed timing coverage."
- "RDNA4 timing is benchmark-grade."
- "The v1.33 release is leaderboard-ready."
- "The timing evidence proves paper parity."
