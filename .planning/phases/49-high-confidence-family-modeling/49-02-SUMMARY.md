---
phase: 49-high-confidence-family-modeling
plan: 02
subsystem: scoring
tags: [solar, linear-projection, evidence, amd-sol]

requires:
  - phase: 49-high-confidence-family-modeling
    plan: 01
    provides: Group-local formula, byte, and bound evidence records in SolarSemanticGroupEvidence
provides:
  - Linear projection estimates preserve first-class family identity while reusing GEMM FLOP formulas
  - Linear projection SOLAR groups include formula, byte, and AMD SOL-style bound evidence
  - Linear projection subroles expose visible input, weight/RHS, bias, and output tensors
affects: [49-high-confidence-family-modeling, 51-sidecar-coverage-and-score-guards]

tech-stack:
  added: []
  patterns: [TDD, sidecar-only evidence, GEMM-compatible projection formulas]

key-files:
  created:
    - .planning/phases/49-high-confidence-family-modeling/49-02-SUMMARY.md
  modified:
    - src/sol_execbench/core/scoring/amd_bound_estimates.py
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_amd_bound_estimates.py
    - tests/sol_execbench/test_solar_derivation_evidence.py

key-decisions:
  - "Linear projection remains op_family=linear_projection while using formula_kind=gemm_flops and formula=2*M*N*K."
  - "Supported GEMM-compatible estimates record axis_source=tensor_shapes so SOLAR confidence can distinguish complete shape provenance from incomplete metadata."
  - "Visible linear projection bias tensors are represented as group-local subrole evidence without changing canonical schemas or score eligibility."

requirements-completed: [DERIVE-06, MODEL-01, MODEL-02, MODEL-05]

duration: 4min
completed: 2026-05-23
---

# Phase 49 Plan 02: Linear Projection Formula Evidence Summary

**Linear projection now has first-class SOLAR family evidence while reusing GEMM-compatible FLOP formulas.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-23T06:39:25Z
- **Completed:** 2026-05-23T06:43:10Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added regression coverage proving complete `OpFamily.LINEAR_PROJECTION` estimates serialize `op_family="linear_projection"` with `formula_kind="gemm_flops"`, `formula="2*M*N*K"`, explicit M/N/K inputs, and dtype-aware byte totals.
- Added degraded projection coverage proving missing output shape or unknown dtype keeps empty formula inputs where dimensions are incomplete and records missing byte evidence warnings.
- Added actual derived SOLAR sidecar coverage for linear projection formula, byte, and bound evidence inside `SolarSemanticGroupEvidence`.
- Added bias subrole extraction for visible linear projection bias inputs while preserving existing `input`, `weight_or_rhs`, and `output` subroles.

## Task Commits

1. **Task 49-02-01 RED tests:** `35d2e5c` (`#49 - Add linear projection estimate tests`)
2. **Task 49-02-01 implementation:** `6d61743` (`#49 - Preserve linear projection GEMM formulas`)
3. **Task 49-02-02 RED tests:** `18385ff` (`#49 - Add linear projection sidecar evidence tests`)
4. **Task 49-02-02 implementation:** `fa0ceb7` (`#49 - Attach linear projection bias sidecar evidence`)

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - Adds `axis_source="tensor_shapes"` to supported GEMM-compatible estimates, including linear projection, without introducing a projection-specific formula kind.
- `src/sol_execbench/core/scoring/solar_derivation.py` - Adds visible bias input subrole evidence for linear/GEMM-style groups.
- `tests/sol_execbench/test_amd_bound_estimates.py` - Adds complete and incomplete linear projection estimate tests.
- `tests/sol_execbench/test_solar_derivation_evidence.py` - Adds linear projection sidecar formula/byte/bound evidence and bias subrole tests.
- `.planning/phases/49-high-confidence-family-modeling/49-02-SUMMARY.md` - Records execution outcome.

## Decisions Made

- Kept `_gemm_estimate()` as the sole formula source of truth for both GEMM and linear projection.
- Kept `formula_kind="gemm_flops"` for linear projection and did not introduce `linear_projection_flops`.
- Kept formula, byte, and bound evidence nested in `SolarSemanticGroupEvidence` per the Plan 49-01 contract.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical evidence] Added tensor-shape axis provenance for supported GEMM-compatible estimates**
- **Found during:** Task 49-02-01
- **Issue:** Complete linear projection estimates had formula inputs and bytes but no `axis_source`, which caused SOLAR classification to treat complete projection evidence as incomplete.
- **Fix:** Set `axis_source="tensor_shapes"` on successful `_gemm_estimate()` results. Incomplete projection estimates still leave `axis_source` absent and degrade.
- **Files modified:** `src/sol_execbench/core/scoring/amd_bound_estimates.py`
- **Commit:** `6d61743`

## Issues Encountered

None.

## Known Stubs

None. Stub scan found only intentional empty fixture/test dictionaries and degraded estimate paths where missing formula inputs are required behavior.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -k "linear or projection or gemm" -n 0 -x` - passed, 2 selected.
- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "linear or projection or formula or byte or bound" -n 0 -x` - passed, 15 selected.
- `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py -k "linear or projection or formula or byte or bound" -n 0` - 33 passed, 23 deselected.
- `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` - 73 passed.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py` - passed.

## Self-Check: PASSED

- Created summary file exists.
- Task commits `35d2e5c`, `6d61743`, `18385ff`, and `fa0ceb7` exist.
- Required files were modified and verification passed.

## User Setup Required

None.

## Next Phase Readiness

Plan 49-03 can build on the same group-local formula, byte, and bound evidence path for the next high-confidence family without changing public schemas or AMD-native score eligibility.

---
*Phase: 49-high-confidence-family-modeling*
*Completed: 2026-05-23*
