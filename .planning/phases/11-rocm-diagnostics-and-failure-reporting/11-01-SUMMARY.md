# Phase 11 Summary: ROCm Diagnostics and Failure Reporting

**Completed:** 2026-05-22
**Plan:** 11-01-PLAN.md
**Status:** Complete

## Changes

- Added `src/sol_execbench/core/diagnostics.py` with internal stage diagnostics,
  ROCm tool checks, gfx classification, local gfx detection, and profiler
  readiness routing.
- Added `src/sol_execbench/core/reporting.py` with pure trace summary helpers.
- Added focused tests for diagnostics and trace reporting.

## Verification

```bash
uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py
```

Included in focused v1.2 test run: 16 passed.
