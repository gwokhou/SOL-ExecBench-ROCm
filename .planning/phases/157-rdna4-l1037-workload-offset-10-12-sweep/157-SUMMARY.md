---
status: complete
---

# Phase 157 Summary

Phase 157 continued the `L1/037_flux_feedforward_gelu_approximate` workload
offset sweep for offsets 10-12.

## Completed

- Ran bounded real RDNA4 `rocprofv3` workload slices for offsets 10, 11, and
  12.
- Confirmed all three offsets collect kernel activity and pass their diagnostic
  single-workload slices.

## Outcome

Offsets 0-12 are individually profiler-compatible. Slow slices include offsets
6, 9, and 11. The next useful step is to sweep offsets 13-15, then attempt a
full-problem profiler replacement with an appropriately larger timeout.
