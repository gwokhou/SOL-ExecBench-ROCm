---
quick_id: 260531-rdf
slug: add-run-dataset-closure-e2e-gaps
status: completed
created: 2026-05-31T14:14:56Z
---

# Quick Task 260531-rdf: Add run_dataset closure E2E gaps

## Goal

Add focused `requires_rocm` regressions for remaining high-value dataset-runner
closure paths that do not expand hardware validation:

1. Ready-subset workload filtering through `--max-workloads`.
2. Missing selected workload references.
3. Readiness-blocked workloads recorded as `not_attempted`.
4. Stale execution-closure provenance forcing a fresh run.

## Tasks

| Task | Files | Action | Verify | Done |
| --- | --- | --- | --- | --- |
| 1 | `tests/examples/test_rocm_cli_paths.py` | Add reusable ready-subset/readiness helpers for one-workload ROCm dataset runs. | Ruff passes. | done |
| 2 | `tests/examples/test_rocm_cli_paths.py` | Add combined e2e for filtered, missing, and readiness-blocked closure records. | Targeted ROCm test passes. | done |
| 3 | `tests/examples/test_rocm_cli_paths.py` | Add stale provenance e2e that verifies re-run and mismatch diagnostics. | Full `requires_rocm` set passes. | done |

## Scope Boundaries

- No Docker build/run.
- No dependency relock.
- No new hardware target claims.
- ROCm/GPU access is limited to `requires_rocm` pytest execution.
