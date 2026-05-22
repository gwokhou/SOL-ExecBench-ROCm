# Phase 32 Verification: Source-Specific Profiler Timing Workflow

**Date:** 2026-05-22
**Verdict:** passed

## Requirement Coverage

| Requirement | Evidence |
|-------------|----------|
| TIME-01 | `timing_policy.py` selects explicit backends; `collect_source_timing_evidence()` records selected policy and fallback metadata. |
| TIME-02 | `scripts/run_dataset.py --timing-evidence-dir` invokes source-specific profiler collection end-to-end for supported policies. |
| TIME-03 | `tests/sol_execbench/test_rocm_profiler.py` covers kernel rows, HIP runtime rows, missing CSV output, and command failure. |
| TIME-04 | `Rocprofv3TimingEvidence.to_dict()` preserves aggregation rule, backend, trial count, warmup runs, iterations, clock-lock status, GPU architecture, and fallback reason. |

## Commands

```bash
uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_run_dataset_amd_score.py
uv run ruff check src/sol_execbench/core/bench/rocm_profiler.py scripts/run_dataset.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_run_dataset_amd_score.py
```

## Result

Both commands passed.
