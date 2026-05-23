---
phase: 49-high-confidence-family-modeling
plan: 03
subsystem: scoring
tags: [solar, attention, amd-bounds, sidecar-evidence, rocm]
requires:
  - phase: 49-01
    provides: "SolarSemanticGroupEvidence formula, byte, and bound evidence contract"
  - phase: 49-02
    provides: "Linear projection family modeling and GEMM-compatible formula behavior"
provides:
  - "Explicit attention graph recognition with qk, softmax, pv, mask, and output projection subroles"
  - "Attention-specific formula, byte, and AMD bound evidence inside SolarSemanticGroupEvidence"
  - "Deterministic degraded and unscored behavior for partial mask semantics and dynamic axes"
affects: [phase-49, phase-50, solar-derivation, amd-bound-graph, amd-bound-estimates]
tech-stack:
  added: []
  patterns: ["Reference-visible graph annotation", "OperatorWorkEstimate-backed sidecar evidence"]
key-files:
  created:
    - tests/sol_execbench/test_solar_derivation_family_modeling.py
  modified:
    - src/sol_execbench/core/scoring/amd_bound_graph.py
    - src/sol_execbench/core/scoring/amd_bound_estimates.py
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_amd_bound_graph.py
    - tests/sol_execbench/test_amd_bound_estimates.py
key-decisions:
  - "Attention recognition stays inside the existing bound graph, estimate, and internal sidecar stack with no public schema changes."
  - "Direct q/k/v tensor inputs are represented as attention projection subroles when the surrounding QK, softmax, and PV structure is statically visible."
  - "Partial mask tensors degrade attention evidence with mask:semantics and mask:sparsity rather than fabricating mask semantics."
patterns-established:
  - "Attention graph promotion annotates existing nodes with OpFamily.ATTENTION and subrole attributes instead of adding new public carriers."
  - "Attention estimates use OperatorWorkEstimate with attention-specific formula kinds except output projection, which remains GEMM-compatible."
requirements-completed: [DERIVE-01, MODEL-01, MODEL-02, MODEL-05]
duration: 7min
completed: 2026-05-23
---

# Phase 49 Plan 03: Explicit Attention Family Modeling Summary

**Reference-visible dense attention now derives formula-backed SOLAR evidence with deterministic degradation for incomplete mask and dynamic-axis cases.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-23T06:46:02Z
- **Completed:** 2026-05-23T06:53:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Promoted explicit QK-softmax-PV-output attention chains into `attention` graph nodes with subrole attributes and static axis metadata.
- Added attention estimates for QK scores, attention softmax, PV aggregation, scale or mask handling, and GEMM-compatible output projection.
- Kept formula, byte, and AMD bound evidence inside `SolarSemanticGroupEvidence`, preserving public schemas and trace behavior.
- Added tests for supported attention, partial-mask degradation, dynamic-axis unscored behavior, graph metadata, and estimate formulas.

## Task Commits

1. **Task 49-03-01/02 RED: Attention family modeling tests** - `46d6a7e` (`#49 - Add attention family modeling tests`)
2. **Task 49-03-01/02 GREEN: Attention graph, estimates, and sidecar evidence** - `4a4d44f` (`#49 - Derive attention formula backed evidence`)

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_bound_graph.py` - Annotates explicit attention chains and dynamic attention-axis evidence.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - Emits attention operation formulas, bytes, bounds, and warnings through `OperatorWorkEstimate`.
- `src/sol_execbench/core/scoring/solar_derivation.py` - Adds attention subroles and family-specific completeness/degradation gates.
- `tests/sol_execbench/test_solar_derivation_family_modeling.py` - Covers supported, degraded, unsupported, and estimate-backed attention behavior.
- `tests/sol_execbench/test_amd_bound_graph.py` - Covers attention graph subrole metadata.
- `tests/sol_execbench/test_amd_bound_estimates.py` - Covers attention formula inputs and byte-producing estimates.

## Decisions Made

- Direct q/k/v inputs are modeled as projection subroles only when the static attention chain is otherwise complete.
- Output projection remains formula kind `gemm_flops` while the containing group remains `attention`.
- Partial mask evidence is degraded with explicit `mask:semantics` and `mask:sparsity` gaps; dynamic sequence axes are unscored.

## Deviations from Plan

### Auto-fixed Issues

None - no unplanned correctness fixes were required.

**Total deviations:** 0 auto-fixed.
**Impact on plan:** Plan scope was preserved.

### Process Adjustments

- Commit messages use the repository-required DCO issue format (`#49 - ...`) instead of the plan's suggested conventional examples.
- The two TDD tasks were implemented through one RED commit and one GREEN commit because their graph, estimate, and sidecar assertions share the same attention behavior.

## Issues Encountered

None.

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py -k attention -n 0` - passed, 6 selected.
- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` - passed, 66 tests.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_family_modeling.py` - passed.

## Known Stubs

None.

## Threat Flags

None - the new attention evidence surface is the planned internal sidecar and graph/estimate surface for this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 49 attention modeling is ready for subsequent high-confidence or degraded family work. Linear projection behavior from 49-02 remains covered by the wave gate.

## Self-Check: PASSED

- Verified created and modified files exist.
- Verified task commits exist: `46d6a7e`, `4a4d44f`.

---
*Phase: 49-high-confidence-family-modeling*
*Completed: 2026-05-23*
