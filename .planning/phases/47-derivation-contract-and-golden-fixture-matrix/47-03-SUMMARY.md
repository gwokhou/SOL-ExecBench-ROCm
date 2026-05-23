---
phase: 47-derivation-contract-and-golden-fixture-matrix
plan: 03
subsystem: tests
tags: [solar, fixtures, attention, moe]

provides:
  - Attention positive/degraded/unsupported fixtures
  - MoE positive/degraded/taxonomy-only fixtures
affects: [phase-47, phase-49, phase-50]
requirements-completed: []
duration: 2min
completed: 2026-05-23
---

# Phase 47 Plan 03: Attention And MoE Fixture Batch Summary

Created six loader-valid JSON fixtures for `attention` and `moe`.

## Commits

- `990282f test(47-03): add attention and MoE SOLAR fixtures`

## Verification

Passed:

```bash
uv run pytest tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x
```

Result: `7 passed`.

## Notes

- The batch contributes `partial`, `dynamic`, and `taxonomy_only` degraded/negative coverage.
- No production scoring, extraction, or modeling code changed.
