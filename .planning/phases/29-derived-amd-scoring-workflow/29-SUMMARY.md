# Phase 29 Summary: Derived AMD Scoring Workflow

**Completed:** 2026-05-22
**Status:** Complete
**Requirements:** SCORE-01, SCORE-02, SCORE-03, SCORE-04

## What Changed

- Added trace-based AMD-native scoring workflow helpers in `amd_score.py`.
- Preserved trace, timing, SOL-bound, baseline, and hardware-model evidence
  references in workload score reports.
- Added dataset runner opt-in `--amd-score-report <path>` for derived suite
  score JSON output.
- Added guarded unscored states for missing timing, baseline, or SOL-bound
  evidence.
- Documented the derived report workflow in `docs/analysis.md`.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py` - passed
- `uv run ruff check src/sol_execbench/core/scoring/amd_score.py scripts/run_dataset.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py` - passed

## Compatibility

Canonical trace JSONL, public trace schemas, primary `sol-execbench` defaults,
and `sol_score()` were not changed. Dataset score reports are opt-in derived
artifacts.
