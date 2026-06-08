---
status: complete
---

# Phase 159 Summary

Phase 159 implemented workload-sharded profiler-backed timing aggregation so
large RDNA4 problems can be closed through bounded workload profiler evidence
instead of unsafe full-problem profiler sessions.

## Completed

- Added `--workload-sharded` mode to the RDNA4 profiler timing batch runner.
- Added deterministic workload profiler manifests with expected workload ids,
  per-workload status, trace paths, kernel rows, timing totals, and retryable
  failure metadata.
- Added complete-only problem-level aggregation sidecars.
- Added `--workload-slice-timing-dir` and `--workload-sharded-import-only` so
  existing real profiler slices can be promoted without rerunning long
  workload sweeps.
- Preserved existing full-problem, workload-offset, blocked-sidecar, skip-list,
  and resume behavior.
- Added CPU-safe tests for complete aggregation, resume, imported slices,
  partial aggregation, and coverage acceptance.

## Real RDNA4 Outcome

`L1/037_flux_feedforward_gelu_approximate` was promoted from blocked/partial
evidence to full problem-level `profiler_backed` evidence by importing all 16
existing real `rocprofv3` workload slices.

Final evidence:

- Batch summary:
  `out/rdna4-profiler-workload-aggregate-20260608-v2/batch-summary.json`
- Manifest:
  `out/rdna4-profiler-workload-aggregate-20260608-v2/workload-manifests/L1/037_flux_feedforward_gelu_approximate.workload-profiler-manifest.json`
- Aggregate sidecar:
  `out/rdna4-profiler-workload-aggregate-20260608-v2/timing/L1/037_flux_feedforward_gelu_approximate.timing.json`
- Coverage summary:
  `out/rdna4-profiler-workload-aggregate-20260608-v2/coverage/coverage-summary.json`

The v2 coverage summary reports 61/235 profiler-backed problems, 5
profiler-blocked problems, 9 partial profiler-backed problems, 46 fallback
timing problems, and 114 readiness-blocked problems.

## Claim Boundary

This closes the structural OOM/timeout issue for aggregatable workload-sharded
profiler timing evidence. It does not upgrade public RDNA4 timing authority,
score authority, paper parity, or leaderboard claims.
