# Phase 166 Context: RDNA4 Rocprof Timing Closure

## Goal

Replace remaining eligible fallback timing with authoritative `rocprofv3`
kernel activity timing for the included RDNA4 denominator.

## Depends On

- Phase 165: RDNA4 coverage recompute

## Scope

- Target eligible included workloads only.
- Keep PyTorch/device-event fallback visible but non-authoritative.
- Preserve long-run controls, bounded retries, and checkpointed artifacts.

## Primary Deliverable

- Profiler timing sidecars with `profiler_collected=true` and `rocprofv3`
  kernel rows for eligible included workloads.
