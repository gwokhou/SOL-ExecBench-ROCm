---
quick_id: 260613-sp3
slug: readiness-blocked-quant-readiness-blocke
status: complete
completed_at: 2026-06-13T20:58:00+08:00
---

# Summary

Closed the `readiness_blocked: Quant` and `readiness_blocked: FlashInfer`
classification gap in the RDNA4 validation overlay.

## Changes

- Refined CUDA runtime hint classification so local compatibility labels such as
  `CuBLASRefBlockwiseGemm()` constructor calls are false positives, while true
  imports/calls such as `cupy` or lowercase `cublas_*()` remain blockers.
- Refined FlashInfer readiness so migrated PyTorch references are considered
  ready unless the reference directly imports or calls the `flashinfer` runtime.
- Split `needs_hardware_evidence` out of coverage `readiness_blocked` into
  `hardware_evidence_deferred`, which covers migrated low-precision paths that
  still need hardware validation evidence.
- Added regression tests for Quant constructor false positives, migrated
  FlashInfer semantic buckets, and hardware-evidence-deferred coverage status.

## Output

- New overlay: `out/rdna4-readiness-quant-flashinfer-closure-20260613/`
- Result: `readiness_blocked_problems = 0`
- Deferred hardware evidence: `hardware_evidence_deferred_problems = 19`
- Ready but missing profiler timing: `ready_missing_profiler_timing_problems = 22`
- Profiler-backed coverage remains `131/235 = 55.7447%`

## Verification

```bash
uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_low_precision_compatibility.py tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_timing_coverage.py -q
```

Result: `53 passed`.
