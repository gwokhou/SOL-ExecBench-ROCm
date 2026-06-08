# Phase 150 Context: RDNA4 Profiler Replacement Classification

## Trigger

Phase 149 proved that the runner can collect real `rocprofv3` kernel activity
rows on RDNA4, but the first full-problem replacement attempt was not fully
replaceable on the current 16 GiB host:

- `L1/002_vae_conv3x3_groupnorm_silu_residual_fused`
- `PASSED`: 19
- `INVALID_REFERENCE`: 1

The strict Phase 149 objective remains blocked because full problem replacement
requires every workload to pass. The next useful phase is therefore
classification, not another blind full-replacement pass.

## Decision

Add Phase 150 to distinguish replacement attempts that are:

- fully `profiler_backed`
- `partial_profiler_backed` with usable `rocprofv3` kernel rows but incomplete
  workload coverage
- `profiler_blocked` when a profiler replacement attempt exists but does not
  produce usable kernel activity evidence

Partial and blocked statuses must stay outside full profiler-backed timing
coverage.

## Claim Boundary

Phase 150 improves accounting and long-run resume behavior. It does not upgrade
RDNA4 timing to score authority, paper parity, leaderboard readiness, or full
235-problem profiler-backed coverage.
