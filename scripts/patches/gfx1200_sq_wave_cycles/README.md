# gfx1200 SQ_WAVE_CYCLES wrapper — DEPRECATED

> **Status: deprecated.** Retained only as a fallback for **standalone**
> `rocprofv3` use outside the SOL ExecBench harness. The harness itself no
> longer needs this wrapper.

## Why it existed

On `gfx1200` (RDNA4) the SQ perf counter `SQ_WAVE_CYCLES` (event 24) reads
exactly zero under the default `AUTO` power policy, because the dVFS
shader-clock transitions suppress its increment. That corrupts the derived
occupancy/stall metrics (`MeanOccupancyPerCU`, `OccupancyPercent`,
`WAVE_DEP_WAIT`, `WAVE_ISSUE_WAIT`). Holding a stable power state during
collection removes the transitions and the counter accumulates normally.
Background: <https://github.com/ROCm/rocm-systems/issues/8523>.

This wrapper entered `STABLE_PEAK` around a real `rocprofv3` invocation and
restored `AUTO` afterwards, so `SQ_WAVE_CYCLES` was valid for harness profiling.

## Why it is deprecated

The clock lock is now acquired **inside the harness**: the rocprofv3 profiling
path wraps collection in `acquire_clock_lock()` (best-effort, graceful skip),
in `src/sol_execbench/cli/evaluation/command.py::_run_profiled_evaluation`. So
any profiling run through `sol-execbench` already runs under `STABLE_PEAK` and
produces valid `SQ_WAVE_CYCLES` values — no wrapper, no separate sudoers rule
beyond the benchmark clock-lock one, no binary swap.

## When you might still use it

You only need this wrapper if you run `rocprofv3` **manually**, outside the
harness (e.g. a one-off counter collection on the bare GPU). For that case,
install and invoke it as before:

```bash
./scripts/patches/gfx1200_sq_wave_cycles/install.sh
~/.local/bin/rocprofv3-gfx1200-patched --kernel-trace --pmc SQ_WAVE_CYCLES -- <workload>
~/.local/bin/rollback-rocprofv3-gfx1200-patch   # remove when done
```

## Notes

- `STABLE_PEAK` is chosen over AMD's recommended `STABLE_STD` because `STD`
  collapses MCLK on this card and bandwidth-starves large kernels (see issue
  data); `STABLE_PEAK` keeps a representative high SCLK/MCLK mix.
