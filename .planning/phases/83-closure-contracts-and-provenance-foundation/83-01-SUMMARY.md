---
phase: 83-closure-contracts-and-provenance-foundation
plan: 01
subsystem: dataset-sidecars
tags:
  - execution-closure
  - provenance
  - deterministic-serialization
dependency_graph:
  requires:
    - src/sol_execbench/core/dataset/checksums.py
  provides:
    - src/sol_execbench/core/dataset/execution_closure.py
  affects:
    - tests/sol_execbench/test_execution_closure_contract.py
tech_stack:
  added:
    - Pydantic execution closure sidecar models
  patterns:
    - stable_json_checksum
    - sorted-key JSON sidecar serialization
key_files:
  created:
    - src/sol_execbench/core/dataset/execution_closure.py
    - tests/sol_execbench/test_execution_closure_contract.py
  modified: []
decisions:
  - Execution closure remains a sidecar-only sol_execbench.execution_closure.v1 contract.
  - Provenance mismatch helpers expose diagnostics without enforcing runner reuse policy in this phase.
metrics:
  completed_date: 2026-05-31
  duration: in-progress
---

# Phase 83 Plan 01: Execution Closure Contract Summary

Strict deterministic execution closure sidecar contract with typed status, reason, totals, checksum, and provenance comparison helpers.

## Completed Tasks

| Task | Description | Status |
|------|-------------|--------|
| 1 | Specified CPU-safe closure contract tests | Complete |
| 2 | Implemented typed execution closure helpers | Complete |

## Implementation Notes

- Added `ExecutionClosureStatus`, `ExecutionClosureReasonCode`, record, provenance, totals, claim-boundary, mismatch, and report models.
- Added deterministic report construction that sorts records by `problem_id`, `row_index`, `workload_uuid`, then `closure_status`.
- Added stable report checksum generation that excludes the checksum field itself.
- Added provenance comparison diagnostics for manifest, readiness, ready-subset, workload identity, solution, and evidence requirement mismatches.

## Verification

Passed:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_execution_closure_contract.py -q
```

Result: `6 passed in 1.16s`

## Deviations from Plan

None - plan executed as written.

## Known Stubs

None.

## Self-Check: PASSED

- Found `src/sol_execbench/core/dataset/execution_closure.py`.
- Found `tests/sol_execbench/test_execution_closure_contract.py`.
- Found commit `d77c6fb`.
