---
phase: 02-rocm-schema-and-native-build-layer
plan: 04
subsystem: testing
tags: [rocm, audit, pytest, schema, build]
requires:
  - phase: 02-rocm-schema-and-native-build-layer
    provides: ROCm schema, packager, and build template changes from 02-01 through 02-03
provides:
  - Focused CUDA/NVIDIA residue audit for Phase 2-owned schema/build paths
  - Combined Phase 2 fast verification suite
affects: [phase-verification, future-schema-build-changes]
tech-stack:
  added: []
  patterns: [path-scoped static pytest audit with explicit allowlist reasons]
key-files:
  created:
    - tests/sol_execbench/test_rocm_schema_build_audit.py
  modified:
    - src/sol_execbench/core/data/solution.py
key-decisions:
  - "Audit only six Phase 2-owned schema/build paths."
  - "Keep allowlist entries exact by path and pattern with non-empty reasons."
  - "Allow PyTorch extra_cuda_cflags only as an upstream API keyword for ROCm device flags."
patterns-established:
  - "Static residue audits report path, pattern, line number, and line text for unallowlisted matches."
requirements-completed: [SCFG-01, SCFG-02, BUILD-01, BUILD-02, BUILD-03, BUILD-04]
duration: 2 min
completed: 2026-05-21
---

# Phase 02 Plan 04: ROCm Schema/Build Audit Summary

**Focused pytest audit guards Phase 2 schema/build paths against unallowlisted CUDA/NVIDIA residue**

## Performance

- **Duration:** 2 min
- **Started:** 2026-05-21T13:05:50Z
- **Completed:** 2026-05-21T13:08:02Z
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments

- Added `tests/sol_execbench/test_rocm_schema_build_audit.py` with the exact six audited paths required by the plan.
- Scans Phase 2-owned schema/build files for CUDA/NVIDIA residue patterns and reports path, pattern, line number, and line text for failures.
- Requires every allowlist entry to target an audited path, use a scanned pattern, and provide a non-empty reason.
- Cleaned stale schema binding/dependency docstrings so the audit allowlist stays focused on intentional migration tests and PyTorch API naming.
- Ran the combined Phase 2 suite across schema, packager, build template, and audit tests.

## Task Commits

1. **Task 1: Focused residue audit and schema doc cleanup** - `5b2f14e` (test)
2. **Task 2: Combined Phase 2 verification** - no code commit; verification-only task

**Plan metadata:** pending in this summary commit

## Files Created/Modified

- `tests/sol_execbench/test_rocm_schema_build_audit.py` - Path-scoped CUDA/NVIDIA residue audit with allowlist reason checks.
- `src/sol_execbench/core/data/solution.py` - Removed stale CUDA/NVIDIA binding and dependency docstrings.

## Decisions Made

- The allowlist covers only intentional migration rejection tests and PyTorch’s `extra_cuda_cflags` public API keyword.
- Examples, docs, evaluator, timing, and broad test migration paths remain outside this Phase 2 failing gate.

## Deviations from Plan

### Auto-fixed Issues

None - plan executed exactly as written.

---

**Total deviations:** 0 auto-fixed.
**Impact on plan:** No scope change.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run --no-sync pytest tests/sol_execbench/test_rocm_schema_build_audit.py` - 2 passed.
- `uv run --no-sync pytest tests/sol_execbench/core/data/test_solution.py tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/driver/test_build_ext.py tests/sol_execbench/test_rocm_schema_build_audit.py` - 107 passed.

## Allowlisted CUDA/NVIDIA References

- `src/sol_execbench/core/data/solution.py` and `tests/sol_execbench/core/data/test_solution.py`: legacy schema values such as `cuda_cpp`, `cuda_cflags`, `cutlass`, `cudnn`, and `cublas` remain only in rejection mappings/tests that prove explicit ROCm migration guidance.
- `tests/sol_execbench/core/data/test_solution.py`: `B200` remains only in a test asserting the old NVIDIA hardware target is absent.
- `src/sol_execbench/driver/templates/build_ext.py` and `tests/sol_execbench/driver/test_build_ext.py`: `extra_cuda_cflags` remains only because PyTorch’s extension API uses that keyword for device compiler flags on ROCm too.

## Next Phase Readiness

Phase 2 implementation is ready for phase-level verification.

---
*Phase: 02-rocm-schema-and-native-build-layer*
*Completed: 2026-05-21*
