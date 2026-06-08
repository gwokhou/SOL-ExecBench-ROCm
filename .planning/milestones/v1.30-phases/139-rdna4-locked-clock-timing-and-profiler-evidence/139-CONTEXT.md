# Phase 139 Context

## Goal

Capture RDNA4 timing context for the completed Phase 138 ready-subset
execution, including environment evidence, clock-lock status, `rocprofv3`
profiler evidence, and timing stability interpretation.

## Inputs

- Phase 138 execution closure:
  `out/rdna4-full-dataset/execution_closure.json`
- Phase 138 run summary:
  `out/rdna4-full-dataset/run/summary.json`
- Phase 138 run output directory:
  `out/rdna4-full-dataset/run`
- Ready subset:
  `out/rdna4-full-dataset/ready_subset.json`
- Readiness report:
  `out/rdna4-full-dataset/readiness.json`
- Dataset manifest:
  `out/rdna4-full-dataset/sol-dataset-manifest.json`

## Current Evidence

- `rocprofv3` is available at `/usr/bin/rocprofv3`.
- `rocprofv3 --version` reports ROCm `7.1.1`.
- `rocm-smi` sees one RDNA4 `gfx1200` AMD Radeon Graphics device.
- `rocm-smi` currently reports a low-power warning and auto/idle clock state,
  so timing authority must remain blocked or non-authoritative until clock
  lock evidence is captured or explicitly classified.

## Key Context

- Long-running validation tasks may run for many hours. Poll approximately
  every 10 minutes, checkpoint meaningful progress, and do not terminate
  healthy processes solely due to duration.
- Phase 139 must not upgrade RDNA4 score, leaderboard, paper parity, or timing
  authority merely because Phase 138 traces exist.
- Failed workloads from Phase 138 remain real RDNA4 findings and must stay
  visible in later reports.
- `UV_CACHE_DIR` should be `/home/guohao/.cache/uv`.

## Claim Boundary

Phase 139 can produce timing-quality evidence and blockers. It cannot by
itself complete derived score reports, public claim wording, hosted
leaderboard authority, upstream SOLAR parity, or unrelated CDNA3/CDNA4 claims.
