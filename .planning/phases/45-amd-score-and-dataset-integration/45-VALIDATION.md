---
phase: 45
slug: amd-score-and-dataset-integration
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-23
---

# Phase 45 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_amd_native_score.py -x` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py -x` |
| **Estimated runtime** | ~30 seconds |

## Per-Task Verification Map

| Task ID | Requirement | Automated Command | Status |
| --- | --- | --- | --- |
| 45-01 | SCORE-01, SCORE-02 | `uv run pytest tests/sol_execbench/test_amd_native_score.py -x` | pending |
| 45-02 | SCORE-03, SCORE-04 | `uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py -x` | pending |

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] No ROCm hardware required.
- [x] No primary CLI or canonical trace JSONL change required.
