---
phase: 47-derivation-contract-and-golden-fixture-matrix
plan: 05
subsystem: tests
tags: [solar, fixtures, embedding-positional, linear-projection]

provides:
  - Embedding/positional positive/degraded/unsupported fixtures
  - Linear projection positive/degraded/unsupported fixtures
affects: [phase-47, phase-49]
requirements-completed: []
duration: 2min
completed: 2026-05-23
---

# Phase 47 Plan 05: Embedding/Positional And Linear Projection Fixture Batch Summary

Created six loader-valid JSON fixtures for `embedding_positional` and `linear_projection`.

## Commits

- `85b4641 test(47-05): add embedding and projection fixtures`

## Verification

Passed:

```bash
uv run pytest tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x
```

Result: `7 passed`.

## Notes

- This batch completes all six required TEST-01 fixture families together with Plans 47-03 and 47-04.
- No production scoring, extraction, or modeling code changed.
