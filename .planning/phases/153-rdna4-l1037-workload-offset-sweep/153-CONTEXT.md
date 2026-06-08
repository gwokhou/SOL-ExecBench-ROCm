---
status: complete
---

# Phase 153 Context

Phase 152 proved that `L1/037_flux_feedforward_gelu_approximate` workload offset
0 can collect `rocprofv3` kernel activity as a diagnostic workload slice, while
the full-problem run times out. Phase 153 sweeps additional workload offsets to
identify which slices are profiler-compatible and which trigger timeout or
profiler failure.

Each workload slice is diagnostic evidence only and must not count as full
profiler-backed timing coverage.
