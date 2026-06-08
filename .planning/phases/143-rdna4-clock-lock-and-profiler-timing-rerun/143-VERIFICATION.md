---
phase: 143
title: RDNA4 clock-lock and profiler timing rerun
status: verified
verified_at: 2026-06-08
---

# Phase 143 Verification

## Automated Checks

| Check | Result |
| --- | --- |
| `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/core/bench/test_clock_lock.py -q` | PASS, 35 passed |
| `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/bench/config/device_config.py src/sol_execbench/core/bench/clock_lock.py tests/sol_execbench/core/bench/test_clock_lock.py` | PASS |

## Live Evidence

| Evidence | Result |
| --- | --- |
| `rocm-smi -s` | RDNA4 host reports SCLK level 2 and MCLK level 5 as supported high-performance levels. |
| ROCm systems source review | `rsmi_dev_gpu_clk_freq_set()` uses an allowed-frequency bitmask and switches to manual performance level. |
| Clock-lock live rerun | PASS; all set/query/reset commands returned 0 and verification passed under mask semantics. |
| Timing sidecar rerun | PASS; 121 sidecars generated under `out/rdna4-timing-evidence-v131/timing`. |

## Timing Classification

All 121 v1.31 timing sidecars recorded:

- `profiler_collected=false`
- backend `device_events`
- activity domain `fallback_event_timing`
- reason `selected policy backend is pytorch_profiler, not rocprofv3 kernel activity timing`

This satisfies the requirement to record availability without overclaiming
fallback event timing as `rocprofv3` kernel activity timing.

