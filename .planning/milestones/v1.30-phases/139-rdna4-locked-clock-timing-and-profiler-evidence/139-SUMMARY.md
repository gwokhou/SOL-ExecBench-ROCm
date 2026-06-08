# Phase 139 Summary

## Result

Phase 139 completed RDNA4 timing-context evidence collection on 2026-06-08.

The completed claim is timing-quality and blocker evidence only. It does not
grant benchmark timing authority, score authority, paper parity, leaderboard
authority, or unrelated CDNA3/CDNA4 validation claims.

## Artifacts

- Environment evidence root:
  `out/rdna4-timing-evidence/environment`
- Timing evidence root:
  `out/rdna4-timing-evidence/timing`
- Stability report JSON:
  `out/rdna4-timing-evidence/stability/evaluation-stability.json`
- Stability report Markdown:
  `out/rdna4-timing-evidence/stability/evaluation-stability.md`
- Clock-lock attempt sidecar:
  `out/rdna4-timing-evidence/environment/clock-lock-attempt.json`

The evidence root contains 132 files, including 121 per-problem timing
sidecars.

## Environment Evidence

- `sol-execbench doctor --json` reports PyTorch ROCm available.
- PyTorch version: `2.10.0+rocm7.1`
- HIP version: `7.1.25424`
- Visible device count: 1
- Device name: `AMD Radeon Graphics`
- Device architecture: `gfx1200`
- `rocprofv3 --version` reports ROCm `7.1.1`.
- `rocminfo` reports a GPU agent named `gfx1200` with marketing name
  `AMD Radeon Graphics`.

## Clock-Lock Status

Clock lock is not complete for this evidence set.

- `rocm-smi --setperflevel manual` succeeded.
- `sudo rocm-smi --setsclk 2` was blocked by a sudo password prompt.
- `sudo rocm-smi --setmclk 5` was blocked by a sudo password prompt.
- `sudo rocm-smi --resetclocks` was blocked by a sudo password prompt.
- `rocm-smi --setperflevel auto` succeeded and restored the performance level
  to auto.

The likely infrastructure issue is incomplete sudoers coverage for GPU
clock-lock and unlock commands. The exact commands needing passwordless
coverage are:

- `sudo rocm-smi --setsclk 2`
- `sudo rocm-smi --setmclk 5`
- `sudo rocm-smi --resetclocks`

Until that sudoers gap is fixed and lock evidence is rerun, RDNA4 timing remains
non-authoritative.

## Timing Evidence

The timing pass reused Phase 138 traces and generated 121 per-problem timing
sidecars under `out/rdna4-timing-evidence/timing`.

Timing sidecar distribution:

- Files: 121
- `profiler_collected=false`: 121
- Policy backend: `device_events` for 121
- Fallback reason for all 121:
  `selected policy backend is pytorch_profiler, not rocprofv3 kernel activity timing`

This means the current reference-solution timing pass did not produce
profiler-backed `rocprofv3` kernel activity timing. The missing profiler timing
is explicit and classified rather than silently absent.

## Stability Evidence

The stability report contains 121 workload entries and classifies all of them
as `missing_timing`.

Status totals:

- `stable`: 0
- `noisy`: 0
- `insufficient_samples`: 0
- `missing_timing`: 121
- `clock_unlocked`: 0
- `profiler_overhead_risk`: 0
- `backend_unsupported`: 0

The stability report keeps timing-quality interpretation true while keeping
correctness, score, paper parity, leaderboard, native-host validation, and
new-hardware validation authority false.

## Follow-Up

- Fix sudoers coverage for the exact clock-lock/reset commands above.
- Rerun Phase 139 clock-lock evidence after sudoers is fixed.
- Collect profiler-backed timing on workloads whose solution source policy can
  route to `rocprofv3` kernel activity timing, or document that PyTorch
  reference solutions remain event-timing/fallback-only.
