---
status: complete
---

# Phase 156 Summary

Phase 156 continued the `L1/037_flux_feedforward_gelu_approximate` workload
offset sweep for offsets 7-9.

## Completed

- Ran bounded real RDNA4 `rocprofv3` workload slices for offsets 7, 8, and 9.
- Confirmed all three offsets collect kernel activity and pass their diagnostic
  single-workload slices.

## Outcome

Offsets 0-9 are individually profiler-compatible. Slow slices now include
offset 6 and offset 9, both requiring a timeout above 300 seconds. The next
useful step is to continue offsets 10-15 before attempting a full-problem
profiler replacement.
