---
status: passed
phase: 29
---

# Phase 29 Verification: Derived AMD Scoring Workflow

**Verified:** 2026-05-22
**Result:** Passed

## Must-Haves

| Requirement | Result | Evidence |
|-------------|--------|----------|
| SCORE-01 | Passed | Core helpers generate workload score reports from trace objects, SOL artifacts, timing refs, and baseline latency inputs. |
| SCORE-02 | Passed | `scripts/run_dataset.py --amd-score-report <path>` writes an opt-in suite JSON report. |
| SCORE-03 | Passed | Score reports preserve trace, timing, SOL-bound, baseline, and hardware-model refs where available. |
| SCORE-04 | Passed | Missing SOL-bound/timing/baseline evidence yields unscored guarded state; unsupported/unvalidated evidence carries warnings. |

## Commands

```bash
uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py
uv run ruff check src/sol_execbench/core/scoring/amd_score.py scripts/run_dataset.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py
```

Both commands passed.

## Residual Risk

Dataset score reports derive SOL bounds during report generation. Future phases
may replace lightweight refs with persisted timing/SOL-bound artifact manifests.
