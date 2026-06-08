---
status: complete
---

# Phase 152 Summary

Phase 152 added workload-sharded profiler triage for RDNA4 timing replacement.

## Completed

- Added `--workload-offset` to the profiler timing batch runner.
- Added workload slice metadata to replacement sidecars:
  `workload_offset`, `workload_slice_applied`, and `workload_slice`.
- Preserved resume semantics so workload slices remain retryable and cannot be
  mistaken for full-problem classifications.
- Added CPU-safe tests for workload offset metadata and slice retry behavior.
- Ran a bounded real RDNA4 workload-slice profiler run for
  `L1/037_flux_feedforward_gelu_approximate`.

## Outcome

The first `L1/037` workload slice succeeded under `rocprofv3` and produced
kernel activity evidence, proving the full-problem timeout is not because every
workload is unprofileable. The slice remains diagnostic-only and does not
increase full profiler-backed coverage.
