# Phase 31 Plan: Optimized Scoring Baseline Semantics

**Created:** 2026-05-22
**Status:** Ready
**Requirements:** BASE-01, BASE-02, BASE-03, BASE-04

<objective>
Implement release-defined scoring baseline artifacts and wire them into
AMD-native score reports without changing canonical trace JSONL.
</objective>

## Tasks

1. Add scoring baseline artifact data structures, parser, validation, and JSON
   serialization helpers.
2. Extend AMD-native scoring to prefer explicit optimized baseline entries and
   label reference-latency fallback as provisional.
3. Add dataset-runner `--scoring-baseline` integration for `--amd-score-report`.
4. Update documentation and focused tests for baseline source semantics.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py`
- `uv run ruff check src/sol_execbench/core/scoring/amd_score.py src/sol_execbench/core/scoring/baseline_artifact.py scripts/run_dataset.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py`

## Risks

- Confusing reference latency with release-defined optimized baseline timing.
- Accidentally expanding canonical trace schema instead of keeping score reports
  derived.
