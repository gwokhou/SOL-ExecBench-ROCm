---
quick_id: 260613-tc4
slug: rdna4-profiler-reference-patch-override-
status: complete
completed_at: 2026-06-13T21:13:00+08:00
---

# Summary

Added explicit marking for RDNA4 profiler timing evidence collected with a
reference implementation override.

## Changes

- `L2/035_convnextv2_block_with_grn` override metadata is now emitted into
  `replacement_metadata.reference_override` whenever the batch applies the
  scoped RDNA4 reference override.
- Coverage evidence summaries now expose `reference_override`.
- Coverage totals now include `reference_override_timing_problems`.
- The RDNA4 blocker ledger includes `reference_override` for non-passing rows.
- Existing `L2/035_convnextv2_block_with_grn.timing.json` sidecars under `out/`
  were backfilled with the same metadata.

## Current Overlay Result

- `out/rdna4-readiness-quant-flashinfer-closure-20260613/coverage-summary.json`
  now reports `reference_override_timing_problems = 1`.
- The marked problem is `L2/035_convnextv2_block_with_grn`.

## Verification

```bash
uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py tests/sol_execbench/test_rdna4_profiler_timing_coverage.py tests/sol_execbench/test_profiler_timing_coverage.py -q
```

Result: `56 passed`.
