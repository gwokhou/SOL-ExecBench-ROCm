---
phase: 43-operator-flop-byte-movement-modeling
plan: 04
subsystem: scoring
tags: [amd, sol, legacy-adapter, public-contract, guardrails]
requires:
  - phase: 43-02
    provides: GEMM and pointwise rich estimates
  - phase: 43-03
    provides: Axis, movement, and dtype conversion rich estimates
provides:
  - Legacy WorkEstimate adapter from rich operator estimates
  - v1 AMD SOL bound artifact compatibility guardrails
  - Public schema and CLI leakage tests for rich estimate fields
affects: [scoring, amd-sol, public-contract, phase-43]
tech-stack:
  added: []
  patterns: [compatibility adapter, public contract guardrail tests]
key-files:
  created: []
  modified:
    - src/sol_execbench/core/scoring/amd_sol.py
    - src/sol_execbench/core/scoring/amd_bound_estimates.py
    - tests/sol_execbench/test_amd_sol_bounds.py
    - tests/sol_execbench/test_public_contract_guardrails.py
key-decisions:
  - "estimate_work() remains a v1 compatibility API returning WorkEstimate."
  - "Legacy bytes_accessed now adapts from rich total_bytes."
  - "v1 artifacts do not serialize formula_inputs, read_bytes, movement_bytes, or operator_work_estimates."
patterns-established:
  - "Rich estimates are the source of scoring evidence; legacy fallback is explicit and exceptional."
  - "Derived scoring fields stay out of canonical Definition, Workload, Trace, and primary CLI contracts."
requirements-completed: [MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05]
duration: "not recorded"
completed: 2026-05-23
---

# Phase 43-04: Legacy Adapter And Guardrail Summary

**Legacy AMD SOL v1 bounds now consume rich operator estimates while preserving WorkEstimate fields, artifact schema, canonical schemas, and primary CLI behavior.**

## Performance

- **Duration:** not recorded
- **Started:** not recorded
- **Completed:** 2026-05-23T09:57:32+08:00
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Updated `amd_sol.estimate_work()` to build a `BoundGraph`, call `estimate_bound_work()`, and adapt each rich estimate to legacy `WorkEstimate`.
- Preserved old whole-definition estimation as an explicit fallback with fallback rationale.
- Added compatibility tests proving legacy `bytes_accessed` equals rich `total_bytes` and `WorkEstimate` fields remain unchanged.
- Strengthened v1 artifact and public contract guardrails so rich estimate fields do not leak into v1 payloads, canonical schemas, or primary CLI help.

## Task Commits

1. **Legacy SOL adapter and guardrails** - `b4a69c8` (feat)

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_sol.py` - Rich-to-legacy adapter and explicit fallback path.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - Softmax rationale wording aligned with existing v1 labeling.
- `tests/sol_execbench/test_amd_sol_bounds.py` - Legacy adapter, v1 artifact, unsupported visibility, and schema field tests.
- `tests/sol_execbench/test_public_contract_guardrails.py` - CLI and canonical schema guardrails for rich estimate field leakage.

## Decisions Made

Followed the plan: `estimate_bound_work()` is now the evidence source, while `estimate_work()` and `AmdSolBoundArtifact` remain stable v1 compatibility views.

## Deviations from Plan

None - plan executed as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_amd_bound_estimates.py -x` - passed, 24 tests.
- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` - passed, 49 tests.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/amd_sol.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py` - passed.

## Next Phase Readiness

Phase 43 has an integrated rich estimate model and stable v1 adapter ready for milestone-level audit or downstream artifact work.

---
*Phase: 43-operator-flop-byte-movement-modeling*
*Completed: 2026-05-23*
