---
status: complete
---

# Phase 153 Summary

Phase 153 continued the `L1/037_flux_feedforward_gelu_approximate` workload
offset sweep.

## Completed

- Ran bounded real RDNA4 `rocprofv3` workload slices for offsets 1, 2, and 3.
- Preserved each offset in a separate output directory to avoid overwriting
  diagnostic sidecars.
- Confirmed offsets 0-3 all collect kernel activity rows and pass their single
  workload slice.

## Outcome

The first four workload offsets for `L1/037` are profiler-compatible. The
remaining investigation should continue with offsets 4-15 or test whether the
full-problem timeout is aggregate runtime rather than a single failing
workload.
