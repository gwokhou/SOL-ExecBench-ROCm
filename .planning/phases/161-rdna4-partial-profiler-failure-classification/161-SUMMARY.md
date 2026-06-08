---
status: complete
---

# Phase 161 Summary

Phase 161 classified current partial profiler-backed RDNA4 timing targets by
failure mode and closure decision.

## Completed

- Added `scripts/run_rdna4_profiler_partial_failures.py`.
- Added CPU-safe tests in
  `tests/sol_execbench/test_rdna4_profiler_partial_failures.py`.
- Generated a deterministic partial failure ledger with JSON, Markdown, and
  per-decision problem-id lists.
- Corrected classification for sharded aggregate sidecars by reading
  `source_workloads`, so failed attempted workloads are not mistaken for
  missing slices.
- Produced the focused `L1/076_batched_expert_forward` deep-dive row.

## Real Classification Result

Artifact directory:

`out/rdna4-profiler-partial-failure-classification-20260608`

Current partial targets: 10.

Closure decisions:

- `blocked_on_correctness`: 6
- `blocked_on_runtime`: 2
- `blocked_on_mixed_failures`: 2

Failure modes:

- `invalid_reference_only`: 6
- `runtime_error_only`: 2
- `mixed_correctness_runtime`: 2

`L1/076_batched_expert_forward` is classified as `runtime_error_only` with
`RUNTIME_ERROR: 14`, so it is blocked on runtime behavior rather than profiler
closure.

## Claim Boundary

This is classification evidence only. It does not increase profiler-backed
coverage and does not upgrade RDNA4 timing authority claims.
