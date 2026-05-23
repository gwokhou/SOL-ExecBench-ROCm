# Phase 45: AMD Score And Dataset Integration - Patterns

**Date:** 2026-05-23
**Status:** Complete

## Pattern Map

| Planned File | Role | Pattern To Reuse |
| --- | --- | --- |
| `src/sol_execbench/core/scoring/amd_score.py` | V2 score consumption and suite summaries | Existing frozen dataclasses, `to_dict()`, warning constants, evidence refs. |
| `scripts/run_dataset.py` | Optional v2 sidecar emission for derived dataset reports | Existing `--amd-score-report` and `--scoring-baseline` derived workflow. |
| `tests/sol_execbench/test_amd_native_score.py` | Score semantics tests | Existing inline trace/artifact fixtures. |
| `tests/sol_execbench/test_run_dataset_amd_score.py` | Dataset helper tests | Existing temporary problem/output fixtures. |

## Reuse Notes

- Keep `AmdNativeScore.supported` as `score is not None`.
- Keep `AmdNativeSuiteReport.to_dict()` backward-compatible while adding
  optional evidence summary fields.
- Keep score evidence refs under the existing `trace`, `timing`, `sol_bound`,
  `baseline`, and `hardware_model` keys.
- Keep sidecar generation outside primary `sol-execbench` CLI.
