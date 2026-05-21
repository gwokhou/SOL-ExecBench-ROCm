---
phase: 01-rocm-environment-baseline
plan: 04
subsystem: testing
tags: [pytest, cleanup, cuda-removal, rocm]
requires:
  - phase: 01-rocm-environment-baseline
    provides: ROCm replacement Docker dependency tests
provides:
  - ROCm-only Docker dependency test collection
affects: [docker-tests, phase-5]
tech-stack:
  added: []
  patterns: [scoped-test-cleanup]
key-files:
  created: []
  modified:
    - tests/docker/dependencies/
key-decisions:
  - "Remove only superseded Docker dependency smoke files under tests/docker/dependencies/."
  - "Keep broader examples, evaluator tests, and source migrations for later phases."
patterns-established:
  - "Phase 1 cleanup is limited to environment smoke tests."
requirements-completed: [ENV-04, SCFG-03]
duration: 15min
completed: 2026-05-21
---

# Phase 1 Plan 04: Docker Dependency Test Cleanup Summary

**ROCm-only Docker dependency pytest collection with superseded CUDA/NVIDIA smoke tests removed**

## Performance

- **Duration:** 15 min
- **Started:** 2026-05-21T05:20:00Z
- **Completed:** 2026-05-21T05:36:48Z
- **Tasks:** 2
- **Files modified:** 8 removed

## Accomplishments

- Removed CUDA compiler, cuDNN, CUTLASS, cuDNN frontend, CuTe DSL, cuTile, helper kernel, and old import-only Triton smoke tests.
- Kept cleanup scoped to `tests/docker/dependencies/`.
- Verified pytest collection now contains ROCm-focused dependency tests only.

## Task Commits

1. **Task 1-2: Remove superseded CUDA/NVIDIA Docker dependency tests** - `aa4d075` (feat)

## Files Created/Modified

- `tests/docker/dependencies/test_cuda.py` - Removed.
- `tests/docker/dependencies/test_cudnn.py` - Removed.
- `tests/docker/dependencies/test_cutlass.py` - Removed.
- `tests/docker/dependencies/test_cudnn_frontend.py` - Removed.
- `tests/docker/dependencies/test_cutedsl.py` - Removed.
- `tests/docker/dependencies/test_cutile.py` - Removed.
- `tests/docker/dependencies/_cutedsl_kernels.py` - Removed.
- `tests/docker/dependencies/test_triton.py` - Removed.

## Decisions Made

The old `test_triton.py` was removed because `test_triton_rocm.py` now owns Triton dependency coverage for the ROCm baseline.

## Deviations from Plan

None. Cleanup stayed within `tests/docker/dependencies/`.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

Phase 5 can treat the Docker dependency directory as ROCm-specific baseline coverage and focus broader test migration elsewhere.

## Verification

- `python -m pytest -p no:xdist -o addopts='' tests/docker/dependencies --collect-only -q` - PASS, 6 tests collected
- `! find tests/docker/dependencies -maxdepth 1 -type f | grep -E 'test_(cuda|cudnn|cutlass)\\.py'` - PASS
- `! find tests/docker/dependencies -maxdepth 1 -type f | grep -E 'test_(cudnn_frontend|cutedsl|cutile|triton)\\.py|_cutedsl_kernels\\.py'` - PASS

## Self-Check: PASSED

---
*Phase: 01-rocm-environment-baseline*
*Completed: 2026-05-21*
