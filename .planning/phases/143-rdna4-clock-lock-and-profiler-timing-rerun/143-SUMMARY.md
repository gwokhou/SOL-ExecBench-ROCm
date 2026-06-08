---
phase: 143
title: RDNA4 clock-lock and profiler timing rerun
status: completed
completed_at: 2026-06-08
---

# Phase 143 Summary

## Completed

- Updated the RDNA4 `gfx1200` clock preset from `SCLK=1, MCLK=1` to the
  observed high-performance DPM levels `SCLK=2, MCLK=5`.
- Kept the generic AMD Radeon preset aligned with the RDNA4 host behavior so
  runtime device names that omit `gfx1200` still use the same levels.
- Adjusted clock verification for ROCm allowed-frequency mask semantics:
  active-level matching remains strong evidence, while low-power state can
  pass only when the requested SCLK/MCLK level is present in `rocm-smi -s` and
  the set command path has already succeeded.
- Verified live RDNA4 clock-lock and reset commands:
  `setperflevel manual`, `setsclk 2`, `setmclk 5`, `showclocks`, `-s`,
  `resetclocks`, and `setperflevel auto` all returned 0.
- Generated a new v1.31 timing evidence set with 121 timing sidecars under
  `out/rdna4-timing-evidence-v131/timing`.

## Evidence

- Clock-lock evidence:
  `out/rdna4-timing-evidence-v131/clock-lock/clock-lock-live-after-mask-semantics.json`
- Timing summary:
  `out/rdna4-timing-evidence-v131/phase143-summary.json`
- Timing sidecar root:
  `out/rdna4-timing-evidence-v131/timing`

## Results

- Clock sudoers blocker: resolved.
- Clock-lock evidence: verified under ROCm allowed-frequency mask semantics.
- Timing sidecars: 121.
- Profiler-backed `rocprofv3` kernel activity sidecars: 0.
- Fallback timing sidecars: 121, all device-event fallback for PyTorch
  reference timing.

## Boundaries

- Phase 143 closes the clock sudoers/lock evidence blocker for RDNA4 timing
  setup.
- Phase 143 does not upgrade timing to profiler-backed kernel activity timing.
  The timing sidecars remain non-authoritative fallback event timing for public
  benchmark/timing claims.

