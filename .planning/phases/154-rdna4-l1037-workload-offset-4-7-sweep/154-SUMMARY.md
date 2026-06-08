---
status: complete
---

# Phase 154 Summary

Phase 154 continued the `L1/037_flux_feedforward_gelu_approximate` workload
offset sweep.

## Completed

- Ran bounded real RDNA4 `rocprofv3` workload slices for offsets 4, 5, and 6.
- Confirmed offsets 4 and 5 collect kernel activity and pass their diagnostic
  single-workload slices.
- Identified offset 6 as the first single-workload timeout blocker.

## Outcome

Offsets 0-5 are profiler-compatible. Offset 6 timed out after 300 seconds
before producing any `PASSED` trace. The next useful action is either to rerun
offset 6 with a larger timeout to distinguish slow workload from hang, or to
skip offset 6 and continue offset 7+ to complete the blocker map.
