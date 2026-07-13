# Phase 29 Plan: Derived AMD Scoring Workflow

**Created:** 2026-05-22
**Status:** Ready for execution
**Requirements:** SCORE-01, SCORE-02, SCORE-03, SCORE-04

## Goal

Connect trace JSONL, live timing evidence, AMD SOL bounds, and baseline inputs
into derived workload and suite score reports.

## Tasks

### 29-01: Add core workflow helpers

**Files:**
- `src/sol_execbench/core/scoring/amd_score.py`
- `tests/sol_execbench/test_amd_native_score.py`

**Work:**
- Add helpers that score workloads from trace objects plus SOL bound artifacts.
- Preserve trace/timing/SOL-bound/baseline/hardware-model evidence refs.
- Keep missing evidence as guarded unscored state.

### 29-02: Add dataset runner report option

**Files:**
- `scripts/run_dataset.py`
- `tests/sol_execbench/test_run_dataset_amd_score.py`

**Work:**
- Add `--amd-score-report <path>`.
- Generate a single suite JSON report only when the option is provided.
- Use existing per-problem traces and problem definitions/workloads.

### 29-03: Add documentation and guardrail tests

**Files:**
- `docs/internal/analysis.md`
- `tests/sol_execbench/test_amd_native_score.py`

**Work:**
- Document the derived report workflow.
- Test trace immutability and missing evidence warnings.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py`
- `uv run ruff check src/sol_execbench/core/scoring/amd_score.py scripts/run_dataset.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py`

## Requirement Mapping

| Requirement | Task |
|-------------|------|
| SCORE-01 | 29-01 |
| SCORE-02 | 29-02 |
| SCORE-03 | 29-01, 29-02 |
| SCORE-04 | 29-01, 29-03 |
