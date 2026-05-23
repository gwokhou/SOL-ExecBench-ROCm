---
phase: 50-degraded-complex-family-modeling
plan: 03
subsystem: scoring
tags: [solar, regression, public-contracts, amd-native-score]

requires:
  - phase: 50-degraded-complex-family-modeling
    provides: "Plans 50-01 and 50-02 MoE and SSM/Mamba degraded-family evidence"
provides:
  - "Focused Phase 50 regression matrix for MoE and SSM/Mamba sidecar evidence"
  - "Public schema, CLI, trace JSONL, and sidecar parser guardrails for Phase 50 names"
  - "AMD-native score eligibility regression coverage for degraded complex-family evidence"
affects: [phase-50, phase-51, solar-derivation, public-contracts, scoring]

tech-stack:
  added: []
  patterns:
    - "Regression-only closure for internal sidecar evidence"
    - "Public-surface negative assertions for internal derivation names"

key-files:
  created:
    - .planning/phases/50-degraded-complex-family-modeling/50-03-SUMMARY.md
  modified:
    - tests/sol_execbench/test_solar_derivation_family_modeling.py
    - tests/sol_execbench/test_public_contract_guardrails.py
    - tests/sol_execbench/test_solar_derivation_evidence.py
    - tests/sol_execbench/test_amd_bound_estimates.py

key-decisions:
  - "Phase 50 internal formula kinds and warning names remain sidecar-only and absent from public schemas, primary CLI help, and canonical trace JSONL."
  - "Degraded MoE and SSM/Mamba evidence does not add SOLAR sidecar references to AMD-native score eligibility."
  - "MoE and SSM/Mamba are no longer generic out-of-scope estimate families; unsupported estimates use family-specific warnings."

patterns-established:
  - "Complex-family closure tests assert fixture contracts at sidecar level and public non-leakage at contract boundaries."

requirements-completed: [DERIVE-02, DERIVE-04]

duration: 6min
completed: 2026-05-23
---

# Phase 50 Plan 03: Regression and Boundary Closure Summary

**Phase 50 closure through focused MoE/SSM regression gates and public AMD-native boundary guardrails**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-23T08:12:38Z
- **Completed:** 2026-05-23T08:18:13Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Tightened MoE and SSM/Mamba sidecar fixture assertions for exact status, confidence, subroles, missing evidence, warning prefixes, and formula-kind decisions.
- Added public guardrails proving Phase 50 internal formula kinds, warnings, and evidence names stay out of `Definition`, `Workload`, `Trace`, canonical trace JSONL, and primary CLI help.
- Added strict sidecar parser coverage proving Phase 50 formula kinds are values inside sidecar evidence, not schema keys.
- Added AMD-native score eligibility regression coverage for degraded MoE and SSM/Mamba artifacts without adding SOLAR sidecar refs.
- Updated the stale generic out-of-scope family estimate regression now that MoE and SSM/Mamba are implemented Phase 50 families.

## Task Commits

1. **Task 50-03-01: Lock complex-family regression matrix** - `fe23e71` (#50 - Lock complex family regression matrix)
2. **Task 50-03-02: Preserve public and AMD-native boundaries** - `8f4ef27` (#50 - Preserve public and AMD score boundaries)
3. **Task 50-03-03: Run full Phase 49 plus Phase 50 gate** - `39f940a` (#50 - Close Phase 50 regression gate)

## Files Created/Modified

- `tests/sol_execbench/test_solar_derivation_family_modeling.py` - Consolidates fixture expectation helpers and locks MoE/SSM sidecar formula-kind regressions.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Adds Phase 50 internal evidence non-leakage checks and AMD-native degraded score boundary coverage.
- `tests/sol_execbench/test_solar_derivation_evidence.py` - Adds strict parser checks for Phase 50 formula kinds as sidecar values only.
- `tests/sol_execbench/test_amd_bound_estimates.py` - Updates unsupported complex-family estimate regression to the Phase 50 family-specific warning contract.

## Decisions Made

- Kept all work in tests; no source, public schema, CLI, trace, dependency, or score eligibility implementation changed.
- Treated degraded MoE and SSM/Mamba SOL v2 artifacts as still AMD-native scoreable when aggregate evidence is degraded but numeric bounds are complete.
- Preserved Phase 51 scope boundaries by not adding sidecar aggregation, score guard integration changes, report output, or dataset runner behavior.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test Expectation Bug] Corrected SSM/Mamba positive sidecar formula-kind expectation**
- **Found during:** Task 50-03-01
- **Issue:** The initial RED assertion expected per-subrole GEMM/convolution/activation formula kinds in the SSM/Mamba sidecar group, but the implemented sidecar contract records visible subrole byte placeholders plus `ssm_mamba_static_scan_flops`.
- **Fix:** Updated the regression to lock the actual internal sidecar formula-kind sequence.
- **Files modified:** `tests/sol_execbench/test_solar_derivation_family_modeling.py`
- **Commit:** `fe23e71`

**2. [Rule 1 - Test Expectation Bug] Limited required-evidence matching to scored fixture groups**
- **Found during:** Task 50-03-01
- **Issue:** The shared fixture helper over-applied positive required-evidence matching to unsupported custom-scan evidence.
- **Fix:** Kept exact status, confidence, subrole, missing-evidence, and warning assertions for unsupported fixtures while limiting required-evidence subset matching to scored groups.
- **Files modified:** `tests/sol_execbench/test_solar_derivation_family_modeling.py`
- **Commit:** `fe23e71`

**3. [Rule 1 - Stale Regression] Updated generic out-of-scope family estimate expectation**
- **Found during:** Task 50-03-03
- **Issue:** The full gate still expected MoE and SSM/Mamba to emit `unsupported_family:torch.linalg.inv`, but Phase 50 intentionally made them recognized families with specific unsupported warnings.
- **Fix:** Updated the regression to lock `unsupported_operator:moe_taxonomy_only` and `unsupported_operator:ssm_custom_scan`.
- **Files modified:** `tests/sol_execbench/test_amd_bound_estimates.py`
- **Commit:** `39f940a`

**Total deviations:** 3 auto-fixed test expectation issues.
**Impact on plan:** No scope broadening; all fixes keep Phase 50 closure as regression and boundary tests only.

## Issues Encountered

- None beyond the auto-fixed test expectation updates documented above.

## Known Stubs

None. Stub-pattern matches in touched files are intentional test fixtures, empty dict/list assertions, or explicit unsupported-evidence cases.

## Threat Flags

None. This plan introduced no new public schema fields, CLI options, trace fields, network endpoints, file access paths, dependencies, score guard integration, or candidate execution surface.

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py -k "moe or ssm or mamba" -n 0 -x` - passed, 19 selected.
- `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_sol_v2.py -n 0 -x` - passed, 68 tests.
- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` - passed, 134 tests.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py` - passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 50 is closed for DERIVE-02 and DERIVE-04 regression coverage. Phase 51 can consume the internal sidecar evidence and warnings for coverage aggregation or score guard integration without public contract drift from Phase 50.

## Self-Check: PASSED

- Summary file exists: `.planning/phases/50-degraded-complex-family-modeling/50-03-SUMMARY.md`.
- Task commits found: `fe23e71`, `8f4ef27`, `39f940a`.
- No tracked files were deleted by task commits.

---
*Phase: 50-degraded-complex-family-modeling*
*Completed: 2026-05-23*
