---
phase: 42-structured-bound-graph-ir
plan: 01
subsystem: scoring
tags: [amd-sol, bound-graph, ir, taxonomy, pytest]
requires:
  - phase: 41-bound-model-contract-and-hardware-artifacts
    provides: AMD hardware confidence/status enums used by graph evidence
provides:
  - Structured bound graph IR dataclasses and serialization
  - Paper-aligned operation family taxonomy
  - Workload-bound declared input/output tensor metadata
  - AST fallback evidence for supported, inexact, and unsupported nodes
affects: [phase-43-operator-modeling, phase-44-bound-artifact-v2]
tech-stack:
  added: []
  patterns: [frozen dataclass IR, explicit to_dict serialization, pytest graph fixtures]
key-files:
  created:
    - src/sol_execbench/core/scoring/amd_bound_graph.py
    - tests/sol_execbench/test_amd_bound_graph.py
  modified:
    - src/sol_execbench/core/scoring/__init__.py
key-decisions:
  - "Keep the rich IR in amd_bound_graph.py instead of expanding amd_sol.py."
  - "Represent unknown calls as explicit unsupported graph evidence."
patterns-established:
  - "BoundGraph/BoundGraphNode/BoundTensor/BoundEdge dataclasses serialize to JSON-like dicts."
  - "OpFamily keeps paper-aligned families separate from legacy amd_sol op_type names."
requirements-completed: [IR-01, IR-02, IR-03]
duration: 25min
completed: 2026-05-23
---

# Phase 42: Structured Bound Graph IR Summary

**Structured AMD bound graph IR with paper-aligned operation families and workload-bound tensor metadata**

## Performance

- **Duration:** 25 min
- **Started:** 2026-05-23T00:31:04Z
- **Completed:** 2026-05-23T00:55:54Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added `amd_bound_graph.py` with `BoundGraph`, `BoundGraphNode`, `BoundTensor`, `BoundEdge`, `BoundTensorRole`, `OpFamily`, and `build_bound_graph()`.
- Added deterministic declared tensor metadata for `Definition` + `Workload` inputs and outputs.
- Added AST fallback evidence that preserves supported, inexact, and unsupported graph nodes with source expressions and rationale.

## Task Commits

Per-task production commits were not created because the worktree already had overlapping uncommitted Phase 41 changes in shared files. Committing only Phase 42 hunks would have risked mixing prior work or reverting user changes.

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_bound_graph.py` - Structured bound graph IR and extraction helpers.
- `tests/sol_execbench/test_amd_bound_graph.py` - Unit tests for taxonomy, metadata, stable IDs, serialization, and unsupported evidence.
- `src/sol_execbench/core/scoring/__init__.py` - Public scoring exports for bound graph API.

## Decisions Made

- Used frozen dataclasses and explicit `to_dict()` methods to match local scoring artifact style.
- Kept formula/SOL analysis out of this plan; conversion hints remain metadata only.

## Deviations from Plan

### Auto-fixed Issues

None.

---

**Total deviations:** 1 execution-protocol deviation (commits deferred due pre-existing overlapping dirty tree).
**Impact on plan:** Implementation and tests are complete; git history needs cleanup before merge/PR.

## Issues Encountered

- Existing Phase 41 changes were present in files also touched by Phase 42. I preserved them and avoided destructive cleanup.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Wave 2 can build on `amd_bound_graph.py` for dynamic trace fallback, edges, and explicit coverage debt.

## Self-Check: PASSED

- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py -x` passed.

---
*Phase: 42-structured-bound-graph-ir*
*Completed: 2026-05-23*

