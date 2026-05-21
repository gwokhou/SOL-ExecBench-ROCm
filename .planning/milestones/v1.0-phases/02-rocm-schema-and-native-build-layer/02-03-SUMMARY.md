---
phase: 02-rocm-schema-and-native-build-layer
plan: 03
subsystem: build
tags: [rocm, hip, pytorch-extension, build-template]
requires:
  - phase: 02-rocm-schema-and-native-build-layer
    provides: Staged hip_cflags from 02-02
provides:
  - HIP/C++ source discovery in build_ext.py
  - HIP compile option pass-through to torch extension loading
  - CUTLASS include path removal from native build defaults
affects: [phase-2-audit, native-extension-build]
tech-stack:
  added: []
  patterns: [PyTorch extension loader with ROCm device flags through extra_cuda_cflags]
key-files:
  created: []
  modified:
    - src/sol_execbench/driver/templates/build_ext.py
    - tests/sol_execbench/driver/test_build_ext.py
key-decisions:
  - "Continue using torch.utils.cpp_extension.load for native extension loading."
  - "Use the PyTorch extra_cuda_cflags keyword for ROCm device compiler flags because it is the public extension API name."
  - "Limit extra include paths to the staging directory by default."
patterns-established:
  - "Build template tests execute the raw template with torch extension loading mocked."
requirements-completed: [BUILD-01, BUILD-03]
duration: 3 min
completed: 2026-05-21
---

# Phase 02 Plan 03: HIP Build Template Summary

**HIP-aware build_ext.py discovers `.hip` and C/C++ sources, reads `hip_cflags`, and preserves the PyTorch extension loader contract**

## Performance

- **Duration:** 3 min
- **Started:** 2026-05-21T13:03:08Z
- **Completed:** 2026-05-21T13:05:50Z
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments

- Updated source discovery to include `.hip`, `.cpp`, `.cc`, `.cxx`, and `.c` while ignoring `.cu`.
- Changed missing-source errors to explicitly mention `HIP/C++`.
- Removed `CUTLASS_DIR` and CUTLASS include path defaults.
- Passed `compile_options.hip_cflags`, `cflags`, and `ld_flags` through `torch.utils.cpp_extension.load`.
- Documented the remaining `extra_cuda_cflags` API keyword as PyTorch’s ROCm-compatible device-compiler flag parameter.

## Task Commits

1. **Tasks 1-2: HIP/C++ source discovery and extension loader flags** - `5aa701f` (feat)

**Plan metadata:** pending in this summary commit

## Files Created/Modified

- `src/sol_execbench/driver/templates/build_ext.py` - HIP/C++ source discovery and HIP compile option pass-through.
- `tests/sol_execbench/driver/test_build_ext.py` - Template execution tests for `.hip`, ignored `.cu`, default/custom `hip_cflags`, and no CUDA linker defaults.

## Decisions Made

- Kept `extra_cuda_cflags` only as the upstream PyTorch API keyword, with an inline comment explaining ROCm use.
- Did not add default ROCm library link flags; `ld_flags` remains opt-in from `solution.json`.

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

- `uv run --no-sync pytest tests/sol_execbench/driver/test_build_ext.py` - 25 passed.

## Next Phase Readiness

Ready for Plan 02-04. The final audit should allowlist only the PyTorch `extra_cuda_cflags` API keyword and its direct tests.

---
*Phase: 02-rocm-schema-and-native-build-layer*
*Completed: 2026-05-21*
