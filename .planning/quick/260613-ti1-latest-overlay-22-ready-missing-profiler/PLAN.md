---
quick_id: 260613-ti1
slug: latest-overlay-22-ready-missing-profiler
status: in_progress
created_at: 2026-06-13T13:14:27.466Z
---

# Quick Task: Close 22 Ready-Missing Profiler Timing Targets

## Goal

Close the latest overlay's 22 `ready_missing_profiler_timing` problems with
RDNA4 profiler-backed timing evidence where locally possible.

## Scope

- Target the 22 rows from
  `out/rdna4-readiness-quant-flashinfer-closure-20260613/blocker-ledger.json`.
- Run profiler batch serially with `--max-workers 1`.
- Keep dynamic estimated timing input preflight enabled.
- Disable HIP runtime tracing if needed to avoid known rocprofv3 instability.
- Record any target that cannot execute locally as a classified blocker.

## Verification

- Rebuild coverage overlay after the batch.
- Report how many targets moved to `profiler_backed`, `reference_oom_blocked`,
  or `profiler_blocked`.
