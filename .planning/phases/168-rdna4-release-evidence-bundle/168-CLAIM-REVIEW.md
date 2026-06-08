# Phase 168 Claim Review

## Supported RDNA4 Claims

The current evidence supports a bounded RDNA4 `gfx1200` release-evidence
statement:

- The RDNA4 ready subset was executed on the recorded host and produced
  auditable trace and closure artifacts.
- The 235-problem denominator is accounted with explicit profiler-backed,
  fallback, readiness-blocked, current-device OOM, and profiler-blocked states.
- 61 of 235 denominator problems have profiler-backed `rocprofv3` timing
  coverage.
- AMD-derived score, AMD SOL v2, and SOLAR derivation sidecars exist as
  project-scoped derived evidence.
- Clock-control and reset commands were exercised on the host.

## Claims That Remain Blocked

The current evidence does not support:

- full 235-problem profiler-backed timing coverage;
- benchmark-grade authoritative timing;
- upstream SOLAR equivalence;
- full paper-parity validation;
- NVIDIA B200 or official leaderboard equivalence;
- score authority;
- CDNA3/MI300X validation;
- CDNA4 validation.

## Timing Authority Review

`rocprofv3` kernel activity timing is present for a subset of RDNA4 problems,
but authoritative timing remains blocked.

Two independent gaps remain:

- Coverage gap: only 61 of 235 denominator problems are profiler-backed.
- Host-policy gap: Phase 167 proved clock-control and reset paths, but did not
  prove stable maximum observed SCLK/MCLK during the benchmark window.

Therefore RDNA4 timing may be described as profiler-backed for the scoped
subset, and fallback timing may be described as diagnostic, but the release
bundle must not claim benchmark-grade timing authority.

## Denominator Review

The accepted Phase 165/166 recompute reports:

- denominator: 235 problems;
- profiler-backed: 61 problems;
- fallback timing: 45 problems;
- current-device OOM blocked: 13 problems;
- profiler blocked: 2 problems;
- readiness blocked: 114 problems.

The current-device blocker classes remain accounted in the denominator but do
not count as complete profiler-backed timing or full validation pass evidence.

## Release Wording

Safe wording:

> RDNA4 `gfx1200` has a release evidence bundle covering bounded execution,
> denominator accounting, partial profiler-backed timing, AMD-derived sidecars,
> blocker ledgers, checksums, and clock-control/reset evidence. Timing remains
> non-authoritative because full profiler-backed coverage and stable observed
> clock lock are not complete.

Avoid wording:

- "RDNA4 is fully validated."
- "Timing is benchmark-grade."
- "SOLAR parity is proven."
- "Leaderboard-ready."
- "CDNA3/MI300X or CDNA4 validated."
