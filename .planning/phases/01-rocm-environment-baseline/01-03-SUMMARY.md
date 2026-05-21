---
phase: 01-rocm-environment-baseline
plan: 03
subsystem: testing
tags: [pytest, rocm, hip, pytorch, triton]
requires:
  - phase: 01-rocm-environment-baseline
    provides: ROCm Docker baseline and ROCm Python dependency declarations
provides:
  - ROCm runtime tool smoke tests
  - HIP compiler smoke test
  - PyTorch ROCm and Triton ROCm smoke tests
  - Selected ROCm library discovery test
affects: [docker-tests, phase-3, phase-5]
tech-stack:
  added: [pytest, hipcc, rocminfo, rocprofv3]
  patterns: [bounded-subprocess-smoke-tests, explicit-missing-rocm-messages]
key-files:
  created:
    - tests/docker/dependencies/test_rocm_runtime.py
    - tests/docker/dependencies/test_hip.py
    - tests/docker/dependencies/test_pytorch_rocm.py
    - tests/docker/dependencies/test_triton_rocm.py
    - tests/docker/dependencies/test_rocm_libraries.py
  modified: []
key-decisions:
  - "Keep ROCm PyTorch device tests on device='cuda' while asserting torch.version.hip."
  - "Use fixed subprocess argument lists and bounded timeouts for ROCm CLI tests."
  - "Report command stdout and stderr on runtime tool failures."
patterns-established:
  - "Docker dependency tests distinguish missing ROCm tools/libraries from Python assertion bugs."
requirements-completed: [ENV-02, ENV-03, ENV-04, SCFG-03]
duration: 35min
completed: 2026-05-21
---

# Phase 1 Plan 03: ROCm Dependency Smoke Test Summary

**ROCm runtime, HIP compiler, PyTorch ROCm, Triton ROCm, and selected library smoke tests**

## Performance

- **Duration:** 35 min
- **Started:** 2026-05-21T05:00:00Z
- **Completed:** 2026-05-21T05:36:48Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added ROCm runtime tool checks for `rocminfo`, `hipcc`, `rocprofv3`, and AMD/ROCm SMI.
- Added a HIP compile-and-run smoke test using `hip/hip_runtime.h`.
- Added PyTorch ROCm and Triton ROCm smoke tests that assert a HIP backend before using GPU APIs.
- Added selected ROCm library discovery checks for `rocblas`, `hipblaslt`, and `MIOpen`.

## Task Commits

1. **Task 1-2: ROCm dependency smoke tests** - `aa4d075` (feat)

## Files Created/Modified

- `tests/docker/dependencies/test_rocm_runtime.py` - ROCm CLI availability and execution checks.
- `tests/docker/dependencies/test_hip.py` - Temporary HIP source compile/run smoke test.
- `tests/docker/dependencies/test_pytorch_rocm.py` - PyTorch ROCm backend and tensor operation smoke test.
- `tests/docker/dependencies/test_triton_rocm.py` - Triton import/version plus ROCm backend visibility smoke test.
- `tests/docker/dependencies/test_rocm_libraries.py` - Selected ROCm library discovery checks.

## Decisions Made

The PyTorch and Triton tests intentionally use PyTorch's `torch.cuda` namespace because ROCm PyTorch exposes HIP devices through that API. `torch.version.hip` is the proof that the wheel is ROCm rather than CUDA.

## Deviations from Plan

The runtime test assertion was tightened after host validation so failed commands include both stdout and stderr. This makes `/dev/kfd` and other environment issues visible in pytest output.

## Issues Encountered

Host runtime execution of `test_rocm_runtime.py` fails because `rocminfo` reports `/dev/kfd` is missing in the current shell environment. `test_hip.py` and `test_rocm_libraries.py` pass on the host. PyTorch/Triton runtime tests were not executed locally because `uv run` would download multi-gigabyte ROCm wheels; they are intended for the Docker/ROCm environment after `uv sync --frozen`.

## User Setup Required

Runtime validation requires an environment with AMD GPU device nodes, especially `/dev/kfd` and `/dev/dri`, plus the locked ROCm Python environment installed.

## Next Phase Readiness

Phase 3 can build on these tests to port runtime/timing behavior, and Phase 5 can broaden hardware validation across RDNA 4 and CDNA 3.

## Verification

- `python -m pytest -p no:xdist -o addopts='' tests/docker/dependencies --collect-only -q` - PASS, 6 tests collected
- `python -m pytest -p no:xdist -o addopts='' tests/docker/dependencies/test_rocm_runtime.py tests/docker/dependencies/test_hip.py -q` - PARTIAL, HIP passed and runtime failed because `/dev/kfd` is absent
- `python -m pytest -p no:xdist -o addopts='' tests/docker/dependencies/test_rocm_libraries.py -q` - PASS
- `python -m py_compile tests/docker/dependencies/test_rocm_runtime.py tests/docker/dependencies/test_hip.py tests/docker/dependencies/test_pytorch_rocm.py tests/docker/dependencies/test_triton_rocm.py tests/docker/dependencies/test_rocm_libraries.py` - PASS

## Self-Check: PASSED WITH ENVIRONMENT-LIMITED RUNTIME VALIDATION

---
*Phase: 01-rocm-environment-baseline*
*Completed: 2026-05-21*
