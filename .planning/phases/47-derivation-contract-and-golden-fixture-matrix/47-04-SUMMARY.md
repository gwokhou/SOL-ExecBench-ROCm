---
phase: 47-derivation-contract-and-golden-fixture-matrix
plan: 04
subsystem: tests
tags: [solar, fixtures, convolution, ssm-mamba]

provides:
  - Convolution positive/degraded/unsupported fixtures
  - SSM/Mamba positive/degraded/unsupported fixtures
affects: [phase-47, phase-49, phase-50]
requirements-completed: []
duration: 2min
completed: 2026-05-23
---

# Phase 47 Plan 04: Convolution And SSM/Mamba Fixture Batch Summary

Created six loader-valid JSON fixtures for `convolution` and `ssm_mamba`.

## Commits

- `79cb427 test(47-04): add convolution and SSM fixtures`

## Verification

Passed:

```bash
uv run pytest tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x
```

Result: `7 passed`.

## Notes

- The batch contributes `missing_metadata`, `dynamic`, and `unsupported` degraded/negative coverage.
- No production scoring, extraction, or modeling code changed.
