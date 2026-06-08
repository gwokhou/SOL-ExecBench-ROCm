---
status: complete
---

# Phase 150 Summary

Phase 150 added replacement classification for RDNA4 profiler timing attempts.

## Completed

- Added `partial_profiler_backed` and `profiler_blocked` status accounting to
  profiler timing coverage.
- Preserved the strict claim boundary: only full workload coverage with
  `rocprofv3` kernel activity rows counts as `profiler_backed`.
- Added replacement metadata for status, workload counts, trace status counts,
  and failure reason.
- Updated batch resume behavior so full-problem classified attempts can be
  skipped while workload-limited smoke artifacts remain retryable.
- Updated the compact coverage summary output to include the new counts.
- Added CPU-safe tests for partial/blocked coverage classification and resume
  semantics.

## Outcome

Replaying the existing real Phase 149 full-problem smoke sidecar now classifies
`L1/002_vae_conv3x3_groupnorm_silu_residual_fused` as
`partial_profiler_backed`. The 235-problem denominator remains fully accounted,
but full profiler-backed timing coverage remains false.
