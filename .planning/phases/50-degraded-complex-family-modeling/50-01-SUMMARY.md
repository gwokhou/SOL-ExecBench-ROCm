---
phase: 50-degraded-complex-family-modeling
plan: 01
subsystem: scoring
tags: [solar, moe, amd-bound-graph, estimates, sidecar]

requires:
  - phase: 49-high-confidence-family-modeling
    provides: high-confidence attention, convolution, embedding, and linear family evidence
provides:
  - MoE graph annotations for router, top-k, dispatch, expert projection, and combine evidence
  - Conservative MoE static and degraded route estimates
  - MoE sidecar subroles, required evidence, missing route metadata, and confidence gates
affects: [phase-50, phase-51, solar-derivation, amd-bound-scoring]

tech-stack:
  added: []
  patterns:
    - source-backed complex-family graph annotations
    - degraded-first route metadata confidence gates

key-files:
  created:
    - .planning/phases/50-degraded-complex-family-modeling/50-01-SUMMARY.md
  modified:
    - src/sol_execbench/core/scoring/amd_bound_graph.py
    - src/sol_execbench/core/scoring/amd_bound_estimates.py
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_amd_bound_graph.py
    - tests/sol_execbench/test_amd_bound_estimates.py
    - tests/sol_execbench/test_solar_derivation_family_modeling.py

key-decisions:
  - "MoE estimates use deterministic formula kinds moe_static_route_flops and moe_dynamic_route_bytes."
  - "Top-k, expert count, token count, and hidden size are included only when visible from parsed constants or tensor shapes."
  - "Taxonomy-only MoE calls remain unscored with unsupported_operator:moe_taxonomy_only."

patterns-established:
  - "MoE dispatch nodes carry source-backed route metadata and explicit missing_route_metadata for degraded dynamic routing."
  - "MoE sidecar confidence adds family-specific aggregate_degraded:moe and aggregate_unscored:moe warnings."

requirements-completed: [DERIVE-02]

duration: 8min
completed: 2026-05-23
---

# Phase 50 Plan 01: Degraded MoE Family Modeling Summary

**Conservative MoE derivation with source-backed static route FLOPs, degraded dynamic-route bytes, and unscored taxonomy-only evidence**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-23T07:48:29Z
- **Completed:** 2026-05-23T07:56:00Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added MoE graph annotation for visible router, top-k, dispatch, expert projection, and combine roles.
- Added deterministic MoE estimate paths for `moe_static_route_flops` and `moe_dynamic_route_bytes`.
- Added sidecar MoE subroles, required evidence, missing route metadata, and degraded/unscored aggregate warnings.
- Preserved Phase 49 attention, convolution, embedding, and score-boundary behavior in the required regression slice.

## Task Commits

1. **Task 50-01-01: Recognize visible MoE graph subroles** - `d5dfeb5` (#50 - Recognize conservative MoE graph evidence)
2. **Task 50-01-02: Emit MoE static and degraded estimates** - `8839283` (#50 - Estimate conservative MoE route work)
3. **Task 50-01-03: Attach MoE sidecar subroles and confidence gates** - `42100ad` (#50 - Attach MoE sidecar confidence evidence)

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_bound_graph.py` - Recognizes visible MoE primitives, records route metadata, and marks taxonomy-only MoE as unsupported.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - Emits static MoE FLOP estimates and degraded visible-byte estimates without route default fabrication.
- `src/sol_execbench/core/scoring/solar_derivation.py` - Adds MoE sidecar subroles, required evidence, confidence gates, and aggregate warnings.
- `tests/sol_execbench/test_amd_bound_graph.py` - Covers MoE graph subroles, dynamic missing route metadata, and taxonomy-only unsupported evidence.
- `tests/sol_execbench/test_amd_bound_estimates.py` - Locks MoE formula kinds and no-fabrication estimate behavior.
- `tests/sol_execbench/test_solar_derivation_family_modeling.py` - Verifies positive, degraded, and unsupported MoE fixture contracts.

## Decisions Made

- Used existing internal sidecar and estimate structures; no public schema, CLI, trace, or score eligibility changes were made.
- Treated `dispatch_and_combine` as a visible compound MoE node that can provide dispatch, expert projection, and combine subroles.
- Kept dynamic routing degraded when top-k or static route cardinality is not visible.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Initial RED graph tests used an input name that conflicted with an axis name; the test fixture was corrected before implementation.
- Positive MoE sidecar grouping initially degraded because router/top-k byte evidence was marked inexact; the estimate confidence was tightened to supported when those primitives have visible source-backed metadata.

## Known Stubs

None.

## Threat Flags

None.

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py -k "moe" -n 0 -x` - passed, 9 selected.
- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_amd_bound_estimates.py -k "moe or attention or convolution or embedding" -n 0` - passed, 18 selected.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_family_modeling.py` - passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 50-02 can add SSM/Mamba modeling using the same graph -> estimate -> sidecar pattern. Plan 51 can consume the explicit degraded/unscored MoE warnings for sidecar coverage and score guards.

## Self-Check: PASSED

- Summary file exists.
- Task commits exist: `d5dfeb5`, `8839283`, `42100ad`.
- No tracked files were deleted by task commits.

---
*Phase: 50-degraded-complex-family-modeling*
*Completed: 2026-05-23*
