---
status: complete
---

# Phase 158 Summary

Phase 158 completed the `L1/037_flux_feedforward_gelu_approximate` workload
offset sweep for offsets 13-15 and closed the serial-sweep investigation.

## Completed

- Ran bounded real RDNA4 `rocprofv3` workload slices for offsets 13, 14, and
  15.
- Confirmed all three offsets collect kernel activity and pass their diagnostic
  single-workload slices.
- Combined with Phases 153-157, confirmed offsets 0-15 are individually
  profiler-compatible.

## Outcome

The problem-level timeout is an aggregate runtime issue, not evidence of an
unprofileable workload. No further serial offset testing is needed for
`L1/037`. The next step should be a full-problem high-timeout attempt or a
workload-slice aggregation design.
