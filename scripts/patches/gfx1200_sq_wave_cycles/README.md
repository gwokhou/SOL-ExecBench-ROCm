# Standalone gfx1200 `SQ_WAVE_CYCLES` wrapper

This wrapper is only for manual `rocprofv3` collection outside the SOL
ExecBench harness. Harness profiling already acquires the same clock lock
internally through `acquire_clock_lock()`; do not install the wrapper for normal
`sol-execbench` commands.

On gfx1200, `SQ_WAVE_CYCLES` can read zero while firmware is changing shader
clocks under the default `AUTO` policy. That invalidates derived occupancy and
stall metrics. The wrapper enters `STABLE_PEAK` for one standalone profiler
invocation and restores `AUTO` afterward. Background:
<https://github.com/ROCm/rocm-systems/issues/8523>.

Use it only for a bounded manual collection:

```bash
./scripts/patches/gfx1200_sq_wave_cycles/install.sh
~/.local/bin/rocprofv3-gfx1200-patched \
  --kernel-trace --pmc SQ_WAVE_CYCLES -- <workload>
~/.local/bin/rollback-rocprofv3-gfx1200-patch
```

`STABLE_PEAK` is used instead of `STABLE_STD` because the latter reduces MCLK
on the validated card and can bandwidth-starve large kernels. Installation does
not grant benchmark authority: preserve exact GPU, clock, profiler, workload,
and cleanup evidence for any manual result.
