---
phase: 50-degraded-complex-family-modeling
plan: 02
subsystem: scoring
tags: [solar, ssm_mamba, amd_bound_graph, operator_estimates, sidecar]

requires:
  - phase: 50-degraded-complex-family-modeling
    provides: "Plan 50-01 MoE degraded-first family modeling patterns"
provides:
  - "SSM/Mamba graph subroles with scan and state-update evidence kept separate"
  - "Static and degraded SSM/Mamba scan estimates with deterministic formula kinds"
  - "SSM/Mamba sidecar subroles, missing recurrence evidence, and aggregate warnings"
affects: [phase-50, phase-51, solar-derivation, scoring]

tech-stack:
  added: []
  patterns: ["Degraded-first compound family modeling through graph -> estimate -> sidecar"]

key-files:
  created: [.planning/phases/50-degraded-complex-family-modeling/50-02-SUMMARY.md]
  modified:
    - src/sol_execbench/core/scoring/amd_bound_graph.py
    - src/sol_execbench/core/scoring/amd_bound_estimates.py
    - src/sol_execbench/core/scoring/solar_derivation.py
    - tests/sol_execbench/test_amd_bound_graph.py
    - tests/sol_execbench/test_amd_bound_estimates.py
    - tests/sol_execbench/test_solar_derivation_family_modeling.py

key-decisions:
  - "SSM/Mamba state_update evidence is emitted only when visible state shape and update parameters exist."
  - "Opaque custom scan calls remain unscored with scan evidence but no fabricated recurrence metadata."
  - "Formula kinds are locked as ssm_mamba_static_scan_flops and ssm_mamba_degraded_scan_bytes."

patterns-established:
  - "Complex family recognition remains internal to bound graph, work estimates, and SOLAR sidecars."
  - "Missing recurrence is represented as degraded evidence, not as guessed state math."

requirements-completed: [DERIVE-04]

duration: 11min
completed: 2026-05-23
---

# Phase 50 Plan 02: SSM/Mamba Degraded Family Modeling Summary

**Conservative SSM/Mamba family evidence with separate scan, state-update, recurrence, formula, and sidecar confidence gates**

## Performance

- **Duration:** 11 min
- **Started:** 2026-05-23T07:59:01Z
- **Completed:** 2026-05-23T08:09:42Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added SSM/Mamba graph annotation for `input_projection`, `depthwise_convolution`, `scan`, `state_update`, `gating`, and `output_projection`.
- Added SSM/Mamba scan estimates with `ssm_mamba_static_scan_flops` for complete recurrence metadata and `ssm_mamba_degraded_scan_bytes` for incomplete recurrence metadata.
- Added sidecar subroles and confidence gates for positive, degraded missing-recurrence, and unsupported custom-scan fixtures.

## Task Commits

1. **Task 50-02-01 RED:** `a01ccac` - `#50 - Add failing SSM Mamba graph tests`
2. **Task 50-02-01 GREEN:** `ac57dc1` - `#50 - Derive SSM Mamba graph roles`
3. **Task 50-02-02 RED:** `7aa9de1` - `#50 - Add failing SSM Mamba estimate tests`
4. **Task 50-02-02 GREEN:** `a0bcc7e` - `#50 - Estimate SSM Mamba scan evidence`
5. **Task 50-02-03 RED:** `f04e9b7` - `#50 - Add failing SSM Mamba sidecar tests`
6. **Task 50-02-03 GREEN:** `bdba2cf` - `#50 - Attach SSM Mamba sidecar evidence`

## Files Created/Modified

- `src/sol_execbench/core/scoring/amd_bound_graph.py` - Adds SSM/Mamba scan-chain annotation while preventing scan-only state-update fabrication.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - Adds SSM/Mamba estimator dispatch and static/degraded scan formulas.
- `src/sol_execbench/core/scoring/solar_derivation.py` - Adds SSM/Mamba sidecar subroles, required evidence, confidence gates, and aggregate warnings.
- `tests/sol_execbench/test_amd_bound_graph.py` - Covers full-chain annotation, missing recurrence, custom scan, and non-SSM conv/projection boundaries.
- `tests/sol_execbench/test_amd_bound_estimates.py` - Covers deterministic static/degraded formula kinds and unsupported custom scan estimates.
- `tests/sol_execbench/test_solar_derivation_family_modeling.py` - Covers positive, degraded, and unsupported SSM/Mamba fixture contracts.

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py -k "ssm or mamba" -n 0 -x` - passed, 10 selected.
- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py -k "moe or ssm or mamba" -n 0` - passed, 19 selected.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_family_modeling.py` - passed.

## Decisions Made

- Kept SSM/Mamba evidence internal to graph, estimate, and SOLAR sidecar layers.
- Required visible state shape and update parameters before emitting `state_update`.
- Kept opaque custom scans unscored with `unsupported_operator:ssm_custom_scan`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test Fixture Bug] Fixed invalid literal dimensions in new Definition shapes**
- **Found during:** Task 50-02-01 RED
- **Issue:** Initial SSM/Mamba test fixtures used literal integers inside `Definition.inputs.*.shape`, but the schema requires axis names.
- **Fix:** Added `one` and `kernel` axes and used those names in convolution-weight shapes.
- **Files modified:** `tests/sol_execbench/test_amd_bound_graph.py`
- **Verification:** The graph RED test then failed for the intended missing SSM/Mamba implementation, and later passed after implementation.
- **Committed in:** `a01ccac`

**Total deviations:** 1 auto-fixed Rule 1 test issue.
**Impact on plan:** No scope change; the correction made the planned tests valid.

## Issues Encountered

None beyond the test fixture correction documented above.

## Known Stubs

None.

## Threat Flags

None. No new public schema, CLI, trace, score eligibility, network endpoint, file access, dependency, or candidate execution surface was introduced.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 50-03 can build on the same degraded-first evidence pattern for public guardrails and regression coverage. Phase 49 behavior and Plan 50-01 MoE behavior remained green in the combined regression gate.

## Self-Check: PASSED

- Summary file exists: `.planning/phases/50-degraded-complex-family-modeling/50-02-SUMMARY.md`.
- Task commits found: `a01ccac`, `ac57dc1`, `7aa9de1`, `a0bcc7e`, `f04e9b7`, `bdba2cf`.
- Stub scan found no blocking placeholders in modified source or tests; matches were normal empty dict/list initializers or explicit unsupported-test assertions.

---
*Phase: 50-degraded-complex-family-modeling*
*Completed: 2026-05-23*
