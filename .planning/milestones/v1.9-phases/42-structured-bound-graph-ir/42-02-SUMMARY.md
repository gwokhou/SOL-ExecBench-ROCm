---
phase: 42-structured-bound-graph-ir
plan: 02
subsystem: scoring
tags: [amd-sol, dynamic-trace, ast-fallback, dataflow, unsupported-evidence]
requires:
  - phase: 42-structured-bound-graph-ir
    provides: BoundGraph IR contract from plan 42-01
provides:
  - Isolated CPU reference trace attempt inside build_bound_graph
  - Producer/consumer edges and deterministic intermediate tensors
  - Explicit dynamic_trace_failed and unsupported_operator warnings
affects: [phase-43-operator-modeling, phase-44-bound-artifact-v2]
tech-stack:
  added: []
  patterns: [isolated derived reference execution, AST fallback, visible coverage debt]
key-files:
  created: []
  modified:
    - src/sol_execbench/core/scoring/amd_bound_graph.py
    - tests/sol_execbench/test_amd_bound_graph.py
key-decisions:
  - "Attempt derived CPU reference execution before AST fallback."
  - "Treat semantic non-coverage as graph evidence, not as dropped work."
patterns-established:
  - "Graph warnings use deterministic strings such as dynamic_trace_failed and unsupported_operator:<name>."
  - "Intermediate tensors use deterministic tmp:op_N:0 IDs."
requirements-completed: [IR-01, IR-02, IR-03]
duration: 15min
completed: 2026-05-23
---

# Phase 42: Structured Bound Graph IR Summary

**Dynamic-trace-first bound graph extraction with AST fallback, dataflow edges, and explicit unsupported evidence**

## Performance

- **Duration:** 15 min
- **Started:** 2026-05-23T00:40:00Z
- **Completed:** 2026-05-23T00:55:54Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Added an isolated CPU reference execution attempt inside `build_bound_graph()`.
- Added deterministic producer/consumer edges and intermediate tensor IDs for common expression graphs.
- Added tests for projection/residual graphs, aliases, tensor methods, chained expressions, tuple outputs, trace failure fallback, dynamic control flow, and unsupported operators.

## Task Commits

Per-task production commits were not created because the worktree already had overlapping uncommitted Phase 41 changes and Wave 1 Phase 42 changes. Changes are present in the working tree and verified by tests.

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_bound_graph.py` - Isolated reference trace attempt, AST fallback dataflow, warnings, and edge construction.
- `tests/sol_execbench/test_amd_bound_graph.py` - Golden tests for dynamic fallback and graph dataflow cases.

## Decisions Made

- Used CPU placeholder execution only for derived metadata; no benchmark timing, correctness, Trace JSONL, CLI, or eval-driver paths are touched.
- Preserved failed trace and unsupported semantics through warnings and graph nodes.

## Deviations from Plan

### Auto-fixed Issues

None.

---

**Total deviations:** 1 execution-protocol deviation (commits deferred due pre-existing overlapping dirty tree).
**Impact on plan:** Functional scope is complete and tested; git history needs cleanup before merge/PR.

## Issues Encountered

- No implementation blockers. The dynamic trace attempt intentionally falls back for untraceable references.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Wave 3 can adapt the rich `BoundGraphNode` records back into legacy `amd_sol.GraphNode` compatibility records.

## Self-Check: PASSED

- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py -x` passed.

---
*Phase: 42-structured-bound-graph-ir*
*Completed: 2026-05-23*

