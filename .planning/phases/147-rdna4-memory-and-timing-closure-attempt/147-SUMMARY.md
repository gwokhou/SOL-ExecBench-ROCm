---
phase: 147
title: RDNA4 memory and timing closure attempt
status: completed
completed: 2026-06-08
---

# Phase 147 Summary

## Result

Phase 147 attempted the two remaining RDNA4 closure items before final handoff:

- Reduced avoidable eval-driver peak memory by replacing unconditional
  reference-output clones with alias-aware stabilization.
- Released correctness outputs before timing so successful correctness rounds do
  not hold extra tensors during latency measurement.
- Added regression coverage proving reference outputs that alias inputs remain
  stable when user code mutates inputs.
- Parsed the v1.31 timing sidecars and confirmed all 121 were collected from
  PyTorch source policies that intentionally route to device-event fallback.
- Reran a known RDNA4 memory-failed workload in an isolated memory-capped unit.

The final v1.31 closure handoff remains:

- `.planning/v1.31-CLOSURE-HANDOFF.md`

## Memory Attempt

Changed files:

- `src/sol_execbench/driver/templates/eval_driver.py`
- `src/sol_execbench/core/bench/rocm_profiler.py`
- `tests/sol_execbench/driver/test_eval_driver.py`
- `tests/sol_execbench/test_rocm_profiler.py`

Targeted RDNA4 retry evidence:

- `out/rdna4-phase147-memory-v131/memory-optimization-attempt.json`
- `out/rdna4-phase147-memory-v131/memory-optimization-attempt.md`

The sampled workload
`L1/026_video_patch_embedding_projection#813ae5c8-9e11-566a-aad4-c9075bf2e616`
still failed with `RUNTIME_ERROR` / HIP OOM. The failure occurs inside the
reference/user function with about 12.74 GiB already allocated by PyTorch, only
2.39 GiB free, and a new 3.16 GiB allocation request. This means the driver
retention optimization is valid but not sufficient to make this 16 GiB RDNA4
workload pass.

## Timing Root Cause

Timing evidence:

- `out/rdna4-phase147-timing-v131/timing-fallback-root-cause.json`
- `out/rdna4-phase147-timing-v131/timing-fallback-root-cause.md`

All 121 v1.31 timing sidecars have `source_type=pytorch`, `backend=device_events`,
and `profiler_collected=false`. `rocprofv3 --version` succeeds on the validation
host, so the root cause is source-language policy routing rather than missing
`rocprofv3` or a CSV parser failure.

Phase 147 also fixed future fallback sidecar metadata so
`selection.policy.reason` preserves the real non-`rocprofv3` policy reason
instead of rewriting PyTorch fallback as profiler unavailability. Existing
v1.31 sidecars remain historical evidence and were not regenerated.

## Accepted Boundaries After Attempt

- RDNA4 timing remains non-authoritative without profiler-backed `rocprofv3`
  kernel activity timing from native HIP, ROCm library, or Triton sources.
- Seven derived-sidecar target problems remain memory blockers under the
  current 32 GiB host and 24G no-swap per-problem cap.
- The 146 failed workload records remain denominator-visible, even though they
  are now classified.
- The result is host-bound to the recorded 16 GiB `gfx1200` validation machine.

## Next-Step Priorities

1. Collect profiler-backed timing evidence.
2. Improve memory efficiency or rerun memory blockers on larger VRAM.
3. Target timeout classes with explicit timeout-overridden reruns.
4. Debug the single `incorrect_numerical` workload.
