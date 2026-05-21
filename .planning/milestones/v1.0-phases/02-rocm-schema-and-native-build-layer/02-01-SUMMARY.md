---
phase: 02-rocm-schema-and-native-build-layer
plan: 01
subsystem: schema
tags: [pydantic, rocm, hip, gfx1200]
requires:
  - phase: 01-rocm-environment-baseline
    provides: ROCm environment baseline and local gfx1200 evidence
provides:
  - ROCm-native solution language enum values
  - Concrete gfx1200 hardware target schema
  - HIP compile option schema and migration validation errors
affects: [problem-packager, build-template, phase-2-audit]
tech-stack:
  added: []
  patterns: [Pydantic before validators for legacy CUDA/NVIDIA migration guidance]
key-files:
  created: []
  modified:
    - src/sol_execbench/core/data/solution.py
    - tests/sol_execbench/core/data/test_solution.py
key-decisions:
  - "Use hip_cpp as the canonical native ROCm C++ language value."
  - "Reject legacy CUDA/NVIDIA schema values instead of accepting compatibility aliases."
  - "Use hip_cflags with minimal default optimization flags and no CUDA linker defaults."
patterns-established:
  - "Legacy CUDA/NVIDIA schema inputs are rejected in before validators with explicit ROCm replacement guidance."
  - "Native ROCm entry points accept .hip and C/C++ suffixes while rejecting .cu."
requirements-completed: [SCFG-01, SCFG-02]
duration: 35 min
completed: 2026-05-21
---

# Phase 02 Plan 01: ROCm Solution Schema Summary

**ROCm-native solution schema with hip_cpp, gfx1200, hip_cflags, and strict CUDA/NVIDIA migration errors**

## Performance

- **Duration:** 35 min
- **Started:** 2026-05-21T12:24:00Z
- **Completed:** 2026-05-21T12:59:39Z
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments

- Replaced CUDA/NVIDIA solution language enum values with `hip_cpp`, `hipblas`, `miopen`, `ck`, and `rocwmma`.
- Replaced `B200` with concrete AMD target `gfx1200` while preserving `LOCAL`.
- Renamed compile options from `cuda_cflags` to `hip_cflags`, removed CUDA linker defaults, and added explicit migration errors.
- Updated schema tests to cover ROCm language acceptance, legacy value rejection, hardware strictness, compile option defaults, and `.hip` entry points.

## Task Commits

1. **Tasks 1-2: ROCm schema, hardware, and compile options** - `ac8b3e9` (feat)

**Plan metadata:** pending in this summary commit

## Files Created/Modified

- `src/sol_execbench/core/data/solution.py` - ROCm solution language, hardware, compile option, and entry suffix validation.
- `tests/sol_execbench/core/data/test_solution.py` - Regression coverage for ROCm schema values and legacy CUDA/NVIDIA rejection.

## Decisions Made

- `hip_cpp` is the canonical native ROCm C++ schema value.
- `hipblas`, `miopen`, `ck`, and `rocwmma` are native ROCm category values for schema/build purposes.
- `gfx1200` is the only explicit AMD target introduced in Phase 2, alongside `LOCAL`.
- `hip_cflags` defaults to `["-O3"]`; `ld_flags` defaults to `[]`.

## Deviations from Plan

### Auto-fixed Issues

None - plan executed exactly as written.

---

**Total deviations:** 0 auto-fixed.
**Impact on plan:** No scope change.

## Issues Encountered

- Initial `uv run pytest` attempted to sync the full Linux ROCm Torch dependency and began downloading a 5 GiB wheel. The run was stopped and replaced with `uv run --no-sync pytest tests/sol_execbench/core/data/test_solution.py`, which verifies this schema-only test without unnecessary dependency hydration.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run --no-sync pytest tests/sol_execbench/core/data/test_solution.py` - 57 passed.

## Next Phase Readiness

Ready for Plan 02-02. `ProblemPackager` can now import ROCm-native schema values and target hardware names.

---
*Phase: 02-rocm-schema-and-native-build-layer*
*Completed: 2026-05-21*
