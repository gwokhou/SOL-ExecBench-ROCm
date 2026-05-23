---
phase: 47-derivation-contract-and-golden-fixture-matrix
plan: 02
subsystem: tests
tags: [solar, fixtures, loader, pytest]

requires:
  - phase: 47-01
    provides: sidecar-only SOLAR derivation contract
provides:
  - Test-side SOLAR derivation fixture JSON loader
  - Loader-only validation tests that pass before fixture JSON files exist
  - Parseable valid negative/degraded fixture expectation validation
affects: [phase-47, phase-48, phase-49, phase-50, phase-51]

tech-stack:
  added: []
  patterns:
    - Test-side stdlib JSON/pathlib loader
    - CPU-only pytest contract validation

key-files:
  created:
    - tests/sol_execbench/solar_derivation_fixtures.py
    - tests/sol_execbench/test_solar_derivation_contract.py
  modified: []

key-decisions:
  - "Kept the loader in the test tier and did not add production scoring or extraction APIs."
  - "Allowed empty fixture directories so loader-only tests can pass before fixture batch plans populate JSON cases."
  - "Validated negative/degraded fixtures as normal parseable payloads rather than exception paths."

patterns-established:
  - "Fixture loading is restricted to sorted `*.json` files under `tests/sol_execbench/fixtures/solar_derivation/`."
  - "Fixture `reference` text is inert and is never executed by the loader."

requirements-completed: []

duration: 2min
completed: 2026-05-23
---

# Phase 47 Plan 02: SOLAR Derivation Fixture Loader Summary

**Test-side loader and loader-only pytest validation for Phase 47 SOLAR derivation fixtures**

## Performance

- **Duration:** 2 min
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Created `tests/sol_execbench/solar_derivation_fixtures.py`.
- Created `tests/sol_execbench/test_solar_derivation_contract.py`.
- Added validation for top-level fixture fields, expectation fields, scope-boundary booleans, family/status/confidence values, and warning prefixes.
- Added tests proving valid degraded/unsupported fixtures are parseable expectations, not exceptions.
- Kept tests CPU-only and independent of ROCm, Docker, compiled extensions, and dataset assets.

## Task Commits

1. **Task 1: Add loader and schema validation unit tests** - `63ec917` (test)

## Files Created/Modified

- `tests/sol_execbench/solar_derivation_fixtures.py` - test-side JSON loader and validation helpers.
- `tests/sol_execbench/test_solar_derivation_contract.py` - loader-only contract tests.

## Decisions Made

- Added an optional `root` parameter to `load_solar_derivation_fixtures()` so tests can validate sorted loading through `tmp_path` without requiring repository fixture files yet.
- Used local test import path setup because `tests/` is not a Python package in this repository.

## Deviations from Plan

None.

## Issues Encountered

- Initial pytest collection failed because `tests` is not an importable package. Fixed by adding the current test directory to `sys.path` before importing the test-side helper.

## Verification

Passed:

```bash
uv run pytest tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x
```

Result: `7 passed`.

## Known Stubs

- No JSON fixtures exist yet; Plans 47-03 through 47-05 populate the fixture matrix.
- Full matrix coverage tests are deferred to Plan 47-06 after fixture files exist.

## Threat Flags

None.

## User Setup Required

None.

## Next Phase Readiness

Plans 47-03, 47-04, and 47-05 can now create fixture JSON batches that are validated by the loader.

## Self-Check: PASSED

- Found loader file: `tests/sol_execbench/solar_derivation_fixtures.py`
- Found test file: `tests/sol_execbench/test_solar_derivation_contract.py`
- Focused pytest passed.

---
*Phase: 47-derivation-contract-and-golden-fixture-matrix*
*Completed: 2026-05-23*
