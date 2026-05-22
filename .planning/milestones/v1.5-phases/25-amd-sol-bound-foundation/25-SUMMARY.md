# Phase 25 Summary: AMD SOL Bound Foundation

**Status:** Complete  
**Completed:** 2026-05-22  
**Code commit:** `e4061d1`  
**Requirements:** SOL-01, SOL-02, SOL-03, SOL-04

## Delivered

- Added `sol_execbench.core.scoring.amd_sol` with derived AMD SOL bound
  artifacts carrying schema version, graph nodes, FLOP/byte work estimates,
  AMD hardware model metadata, per-op bounds, and aggregate bound.
- Implemented conservative graph extraction for supported matmul patterns,
  inexact elementwise patterns, and visible unsupported operations.
- Added built-in `gfx1200` provisional and `gfx942` unvalidated hardware model
  entries, preserving the milestone rule that CDNA3 validation is out of scope.
- Documented that AMD-native scores require an AMD SOL bound artifact before
  reporting.

## Notes

- The implementation is a foundation, not a complete SOLAR clone. Unsupported
  operations remain visible with confidence `unsupported` and zero FLOPs so
  later scoring can guard claims instead of silently inventing evidence.
- SOL and timing evidence remain derived artifacts and do not mutate canonical
  trace JSONL.
