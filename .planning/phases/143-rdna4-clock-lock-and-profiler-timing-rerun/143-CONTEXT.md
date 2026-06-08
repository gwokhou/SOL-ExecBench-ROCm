---
phase: 143
title: RDNA4 clock-lock and profiler timing rerun
---

# Phase 143 Context

Phase 139 left RDNA4 timing non-authoritative because `rocm-smi` clock-lock
and reset commands still required a sudo password. Phase 142 added and verified
command-specific passwordless sudoers coverage for `/usr/bin/rocm-smi`.

Phase 143 reruns the clock-lock evidence and timing sidecar generation after
that sudoers fix. The phase must keep two claim boundaries separate:

- Successful clock command execution and verification can close the sudoers
  blocker.
- PyTorch reference timing sidecars may still fall back to device-event timing;
  those sidecars must not be described as profiler-backed `rocprofv3` kernel
  activity timing.

The user pointed to the ROCm systems source for clock behavior. Local source at
`/home/guohao/CLionProjects/rocm-systems` shows
`rsmi_dev_gpu_clk_freq_set()` accepts a frequency bitmask, limits the set of
allowed frequencies, and switches the device to manual performance level. This
means `rocm-smi --setsclk/--setmclk` is not a guarantee that the current idle
clock line will always show the requested level as active.

