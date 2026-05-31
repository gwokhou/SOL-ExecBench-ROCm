---
phase: 87-amd-sol-solar-bound-sanity-evidence
plan: 02
subsystem: tooling
tags: [amd-bound-sanity, script, guardrails, sidecar, reporting, tdd]
requires:
  - phase: 87-amd-sol-solar-bound-sanity-evidence
    provides: amd_bound_sanity.v1 core report builder and write helpers from Plan 87-01
provides:
  - thin scripts/report_amd_bound_sanity.py wrapper
  - CPU-safe script tests for explicit existing-artifact inputs
  - public contract guardrails for canonical schemas, AMD score contracts, and primary CLI help
affects: [phase-88-docs, public-contract-guardrails, reporting-tooling]
tech-stack:
  added: []
  patterns: [argparse sidecar script, importlib script tests, primary CLI non-exposure guardrails]
key-files:
  created:
    - scripts/report_amd_bound_sanity.py
    - tests/sol_execbench/test_amd_bound_sanity_script.py
  modified:
    - tests/sol_execbench/test_public_contract_guardrails.py
key-decisions:
  - "AMD bound sanity generation is exposed only as a research script, not as a primary sol-execbench CLI option or package entry point."
  - "The script loads only explicitly supplied JSON paths and delegates normalization, checksum, status, and rendering behavior to Plan 87-01 core helpers."
  - "Public guardrails assert the diagnostic sidecar remains out of canonical Definition/Workload/Trace/Solution payloads and AMD-native score contracts."
patterns-established:
  - "Script tests import the script file directly and call main() with monkeypatched argv for deterministic CPU-safe coverage."
  - "Guardrails check both absence from public contracts and presence of negative claim-boundary wording."
requirements-completed: [SANITY-01, SANITY-02, SANITY-03, SANITY-04]
duration: 4min
completed: 2026-05-31
---

# Phase 87 Plan 02: AMD Bound Sanity Script And Guardrails Summary

**Thin AMD bound sanity report script with public contract guardrails proving the artifact stays sidecar-only and diagnostic-only**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-31T11:16:34Z
- **Completed:** 2026-05-31T11:20:59Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added `scripts/report_amd_bound_sanity.py`, a thin argparse wrapper over the Plan 87-01 core helpers.
- Added deterministic script tests that generate JSON/Markdown from explicit local JSON sidecar paths and preserve all diagnostic statuses.
- Extended public contract guardrails to keep bound sanity fields out of canonical schemas, AMD score payloads, and primary `sol-execbench --help`.

## Task Commits

1. **Task 1: Specify script wrapper behavior** - `ade4b2c` (test RED)
2. **Task 2: Implement thin report script** - `170751a` (feat GREEN)
3. **Task 3: Add public contract and claim-boundary guardrails** - `2c95d0f` (test)

## Files Created/Modified

- `scripts/report_amd_bound_sanity.py` - Thin script wrapper that reads explicit JSON sidecar paths and writes deterministic JSON/Markdown reports.
- `tests/sol_execbench/test_amd_bound_sanity_script.py` - CPU-safe script coverage for deterministic output and missing optional evidence gaps.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Guardrails for canonical schema exclusion, AMD score exclusion, primary CLI non-exposure, and visible negative boundaries.

## Decisions Made

- No console script or primary `sol-execbench` CLI wiring was added.
- Missing optional artifact paths are left to the core builder as diagnostic evidence gaps.
- Guardrail coverage validates claim boundaries through report Markdown rather than documentation edits, leaving Phase 88 documentation work separate.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated stale hardware-boundary guardrail wording**
- **Found during:** Task 3 final guardrail verification.
- **Issue:** An existing guardrail expected the old phrase `CDNA 3 / MI300X real-hardware validation`, while current v1.19 requirements use `CDNA 3, MI300X, CDNA 4, or native-host ROCm validation expansion`.
- **Fix:** Updated the assertion to match the current requirements wording while preserving the same deferred-hardware-validation boundary.
- **Files modified:** `tests/sol_execbench/test_public_contract_guardrails.py`
- **Verification:** Focused public contract/script/core/score test command passed.
- **Committed in:** `2c95d0f`

**Total deviations:** 1 auto-fixed bug.
**Impact on plan:** No scope expansion; the fix keeps the user's explicit no-new-hardware-validation constraint enforceable.

## Issues Encountered

- None beyond the stale guardrail wording documented above.

## Known Stubs

None. Stub scan only found argparse defaults and existing test fixture `None` values.

## Threat Flags

None. The new script reads only explicit local JSON paths and does not add network endpoints, auth paths, filesystem scans, Docker/ROCm probes, or trust-boundary changes beyond those in the plan threat model.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_bound_sanity_script.py -q` failed as expected during RED because the script was absent.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_bound_sanity_script.py tests/sol_execbench/test_amd_bound_sanity.py -q` passed: 9 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_bound_sanity_script.py tests/sol_execbench/test_amd_bound_sanity.py tests/sol_execbench/test_amd_native_score.py -q` passed: 67 passed.
- Final combined verification passed: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_bound_sanity.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_sanity_script.py tests/sol_execbench/test_public_contract_guardrails.py -q` (157 passed).

## Self-Check: PASSED

- Created files exist: `scripts/report_amd_bound_sanity.py`, `tests/sol_execbench/test_amd_bound_sanity_script.py`.
- Modified guardrail file exists: `tests/sol_execbench/test_public_contract_guardrails.py`.
- Commits exist: `ade4b2c`, `170751a`, `2c95d0f`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 88 can document the script and add any broader docs/guardrail coverage without changing benchmark execution, primary CLI, Docker, dependencies, or hardware-validation scope.

---
*Phase: 87-amd-sol-solar-bound-sanity-evidence*
*Completed: 2026-05-31*
