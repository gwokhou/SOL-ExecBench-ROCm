---
status: complete
---

# Phase 155 Summary

Phase 155 reran `L1/037_flux_feedforward_gelu_approximate` workload offset 6
with a 900 second timeout.

## Completed

- Ran the offset 6 single-workload profiler slice through bounded
  `systemd-run --user`.
- Confirmed the slice passes and collects `rocprofv3` kernel activity when the
  timeout is raised from 300 to 900 seconds.

## Outcome

Offset 6 is not a hang. It is a slow profiler workload slice with
`kernel_duration_ms=316696.339607` and `1681` parsed kernel rows. The next
useful step is to continue offsets 7-15 or calculate a full-problem timeout
large enough for all 16 workload slices.
