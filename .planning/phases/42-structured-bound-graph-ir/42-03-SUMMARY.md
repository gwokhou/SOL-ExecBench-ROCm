---
phase: 42-structured-bound-graph-ir
plan: 03
subsystem: scoring
tags: [amd-sol, compatibility, public-contract, guardrails]
requires:
  - phase: 42-structured-bound-graph-ir
    provides: BoundGraph IR and extraction path from plans 42-01 and 42-02
provides:
  - amd_sol.extract_graph compatibility adapter over BoundGraphNode
  - Deliberate public scoring exports for the bound graph API
  - Public schema, Trace JSONL, and primary CLI guardrails
affects: [phase-43-operator-modeling, phase-44-bound-artifact-v2, phase-45-score-integration]
tech-stack:
  added: []
  patterns: [compatibility facade, public-contract guardrail tests]
key-files:
  created: []
  modified:
    - src/sol_execbench/core/scoring/amd_sol.py
    - src/sol_execbench/core/scoring/__init__.py
    - tests/sol_execbench/test_amd_bound_graph.py
    - tests/sol_execbench/test_amd_sol_bounds.py
    - tests/sol_execbench/test_public_contract_guardrails.py
key-decisions:
  - "Map rich OpFamily values back to legacy GraphNode.op_type values for existing amd_sol estimators."
  - "Keep AmdSolBoundArtifact v1 schema unchanged in Phase 42."
patterns-established:
  - "New derived APIs are exported only through sol_execbench.core.scoring."
  - "Canonical Definition, Workload, Trace, and primary CLI contracts exclude bound graph fields."
requirements-completed: [IR-01, IR-02, IR-03, IR-04]
duration: 15min
completed: 2026-05-23
---

# Phase 42: Structured Bound Graph IR Summary

**AMD SOL compatibility facade now consumes structured bound graph evidence while preserving public contracts**

## Performance

- **Duration:** 15 min
- **Started:** 2026-05-23T00:45:00Z
- **Completed:** 2026-05-23T00:55:54Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Updated `amd_sol.extract_graph()` to build a minimal workload, consume `build_bound_graph()`, and flatten `BoundGraphNode` to legacy `GraphNode`.
- Exported `BoundGraph`, `BoundGraphNode`, `BoundTensor`, `BoundEdge`, `BoundTensorRole`, `OpFamily`, and `build_bound_graph` through the scoring package.
- Strengthened public-contract tests to reject bound graph leakage into canonical schemas and primary CLI help.

## Task Commits

Per-task production commits were not created because the worktree already had overlapping uncommitted Phase 41 changes in `amd_sol.py`, `__init__.py`, and tests. Changes are present in the working tree and verified by tests.

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_sol.py` - Legacy facade delegation and `OpFamily` to legacy `op_type` mapping.
- `src/sol_execbench/core/scoring/__init__.py` - Deliberate bound graph exports.
- `tests/sol_execbench/test_amd_bound_graph.py` - Public scoring export smoke test.
- `tests/sol_execbench/test_amd_sol_bounds.py` - Existing compatibility suite remains passing.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Guardrails for schema and CLI non-leakage.

## Decisions Made

- `gemm` and `linear_projection` map to legacy `matmul`; `mlp_activation` maps to `activation`; `dtype_conversion` maps to `data_movement`; unsupported remains unsupported.
- `AmdSolBoundArtifact` remains v1 in this phase; v2 sidecars remain Phase 44.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Ruff re-export warning for `HardwareValidationStatus`**
- **Found during:** Wave 3 verification
- **Issue:** Ruff treated the facade import as unused in `amd_sol.py`.
- **Fix:** Marked it as an explicit re-export with `HardwareValidationStatus as HardwareValidationStatus`.
- **Files modified:** `src/sol_execbench/core/scoring/amd_sol.py`
- **Verification:** Ruff check passed.
- **Committed in:** not committed separately due dirty overlap.

---

**Total deviations:** 1 auto-fixed, 1 execution-protocol deviation (commits deferred due pre-existing overlapping dirty tree).
**Impact on plan:** Compatibility and public-contract behavior are verified; git history needs cleanup before merge/PR.

## Issues Encountered

- Existing Phase 41 changes remained in shared files. I preserved them and layered Phase 42 changes on top.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 43 can consume `BoundGraph` as the source of operator formulas and byte/movement evidence.

## Self-Check: PASSED

- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` passed.
- `uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` passed.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_sol.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py` passed.

---
*Phase: 42-structured-bound-graph-ir*
*Completed: 2026-05-23*

