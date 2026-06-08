---
status: planned
---

# Phase 160 Context

## Problem

Phase 159 proved that complete workload-sharded profiler evidence can safely
promote a large blocked target to problem-level `profiler_backed` timing.
The remaining profiler closure work should now target the existing
`partial_profiler_backed` and `profiler_blocked` buckets before expanding to
the larger fallback bucket.

## Decision

Phase 160 will first build a deterministic remaining-target audit. The audit
must identify which targets are candidates for import-only aggregation, which
need missing workload slices, which should be freshly sharded, and which should
remain blocked.

## Claim Boundary

The audit and target list do not upgrade coverage by themselves. A problem only
becomes `profiler_backed` after complete per-workload profiler evidence is
aggregated into a problem-level sidecar.
