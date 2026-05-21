---
phase: 03-rocm-evaluation-timing-and-hardware-introspection
plan: 03
subsystem: clock-env
tags:
  - rocm
  - clock-lock
  - environment
key-files:
  - src/sol_execbench/core/bench/clock_lock.py
  - src/sol_execbench/core/bench/config/device_config.py
  - src/sol_execbench/core/utils.py
  - tests/sol_execbench/core/bench/test_clock_lock.py
metrics:
  tests: "41 passed"
---

# Plan 03-03 Summary

## Changes

- Replaced `nvidia-smi` clock-lock logic with ROCm `rocm-smi` probing, SCLK/MCLK DPM level locking, verification, and best-effort reset.
- Replaced NVIDIA frequency presets with AMD/ROCm DPM-level presets for `gfx1200`, `gfx942`, AMD Radeon, and AMD Instinct names.
- Added `SOL_EXECBENCH_SCLK_LEVEL` and `SOL_EXECBENCH_MCLK_LEVEL` overrides.
- Updated environment snapshots to report `hip` and `rocm` versions when PyTorch exposes `torch.version.hip`.
- Preserved the eval-driver gate that rejects workloads with runtime-error traces when `lock_clocks=True` but `SOL_EXECBENCH_CLOCKS_LOCKED` is not set.

## Verification

- `uv run --no-sync pytest tests/sol_execbench/core/bench/test_clock_lock.py tests/sol_execbench/driver/test_eval_driver.py` -> 41 passed.

## Deviations

None.

## Self-Check: PASSED

Clock locking and environment reporting now use AMD/ROCm names and tooling, and lock success remains required before benchmark timing when requested.
