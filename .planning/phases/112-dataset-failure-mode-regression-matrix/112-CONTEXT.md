# Phase 112 Context: Dataset Failure-Mode Regression Matrix

## Objective

Make dataset-run failure and reuse behavior explicit through a regression
matrix, with documentation that distinguishes CPU-safe policy tests from live
ROCm execution.

## Existing Coverage

- `tests/sol_execbench/test_run_dataset_execution_closure.py` covers ready
  subsets, stale provenance, forced rerun, missing derived evidence, CLI
  no-output, nonzero exit, timeout, and missing selected traces.
- `tests/sol_execbench/test_dataset_run_closure.py` covers reusable core
  helpers and evidence completeness.

## Scope

- Document the dataset failure-mode matrix in user-facing dataset analysis
  docs.
- Add a lightweight doc regression test for the matrix terms and CPU/live ROCm
  boundary.

## Out Of Scope

- New live GPU tests.
- New dataset runner behavior beyond documenting and pinning the existing
  matrix.
