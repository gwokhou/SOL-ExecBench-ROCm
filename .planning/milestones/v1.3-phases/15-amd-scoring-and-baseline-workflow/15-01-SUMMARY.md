# Phase 15 Summary: AMD Scoring and Baseline Workflow

**Completed:** 2026-05-22
**Plan:** 15-01-PLAN.md
**Status:** Complete

## Changes

- Added `sol_execbench.core.baseline` helpers for loading trace JSONL,
  comparing candidates against fastest matching baselines, and formatting text
  or JSON output.
- Added the public `sol-execbench-baseline` command.
- Documented baseline comparison and AMD-native score interpretation guardrails
  in `docs/internal/analysis.md`.
- Added focused tests for classification, CLI output, JSON output, and AMD claim
  warnings.

## Verification

```bash
uv run pytest tests/sol_execbench/test_baseline_comparison.py
```

Result: 4 passed.
