# Phase 45: AMD Score And Dataset Integration - Research

**Date:** 2026-05-23
**Status:** Complete

## Findings

- The current AMD-native score module already has the correct derived-report
  boundary: `AmdNativeScore`, `AmdNativeSuiteReport`, evidence refs, baseline
  source, and canonical-output marker.
- The current score path consumes v1 `AmdSolBoundArtifact`. Phase 44 introduced
  a richer `AmdSolBoundV2Artifact` with aggregate status, artifact warnings,
  coverage, and hardware model references.
- Dataset reporting already happens only when `scripts/run_dataset.py` receives
  `--amd-score-report`, so optional v2 sidecar emission can be scoped there
  without changing canonical trace JSONL or primary CLI behavior.

## Implementation Strategy

- Extend `score_amd_native_workload()` and trace/suite helpers to accept either
  v1 or v2 artifacts.
- For v2 artifacts:
  - use `aggregate_bound.sol_bound_ms` as the numeric bound evidence;
  - if `aggregate_bound.scored` is false, return `score=None`;
  - append v2 artifact warnings and a stable unscored/degraded score warning.
- Add suite evidence summary fields while preserving existing `scored_count`,
  `unscored_count`, and baseline summary fields.
- Update dataset helper to build v2 sidecars and optionally write them to disk
  when a new script-level sidecar directory argument is provided.

## Validation

Focused CPU commands:

- `uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py -x`
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_score.py scripts/run_dataset.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py`
