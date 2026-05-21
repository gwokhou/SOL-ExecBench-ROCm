---
phase: 02-rocm-schema-and-native-build-layer
plan: 02
subsystem: driver
tags: [rocm, hip, gfx1200, packaging]
requires:
  - phase: 02-rocm-schema-and-native-build-layer
    provides: ROCm-native solution schema values from 02-01
provides:
  - ROCm native-language detection in ProblemPackager
  - Local AMD gfx target probing through ROCm tools
  - HIP offload architecture flag injection into staged solution metadata
affects: [build-template, phase-2-audit, native-execution]
tech-stack:
  added: []
  patterns: [fixed-argv subprocess probing, staged solution metadata mutation]
key-files:
  created: []
  modified:
    - src/sol_execbench/driver/problem_packager.py
    - tests/sol_execbench/driver/test_problem_packager.py
key-decisions:
  - "Use rocm_agent_enumerator first, then rocminfo, for LOCAL gfx detection."
  - "Inject --offload-arch flags into hip_cflags rather than requiring solutions to provide them manually."
  - "Leave hip_cflags unchanged when an explicit offload/AMD target flag is already present."
patterns-established:
  - "ProblemPackager owns target-specific compiler flag injection while preserving staged solution.json as the build template contract."
requirements-completed: [BUILD-01, BUILD-02]
duration: 4 min
completed: 2026-05-21
---

# Phase 02 Plan 02: ROCm Packager Target Injection Summary

**ProblemPackager stages HIP/C++ solutions and injects AMD `--offload-arch` flags from explicit or local gfx targets**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-21T12:59:39Z
- **Completed:** 2026-05-21T13:03:08Z
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments

- Replaced native language detection with ROCm schema values: `hip_cpp`, `hipblas`, `miopen`, `ck`, and `rocwmma`.
- Replaced NVIDIA SM/gencode probing with `_get_local_gfx()` using `rocm_agent_enumerator -name` and `rocminfo`.
- Injected `--offload-arch=gfx1200` into staged `compile_options.hip_cflags` for explicit `gfx1200` and mocked `LOCAL` targets.
- Updated packager tests to use HIP sources and validate explicit, local, and pre-existing offload flag behavior.

## Task Commits

1. **Tasks 1-2: ROCm native detection and offload arch injection** - `a16ad9f` (feat)

**Plan metadata:** pending in this summary commit

## Files Created/Modified

- `src/sol_execbench/driver/problem_packager.py` - ROCm native language detection, gfx probing, and offload arch injection.
- `tests/sol_execbench/driver/test_problem_packager.py` - HIP staging, local gfx detection, and offload arch injection tests.

## Decisions Made

- `_get_local_gfx()` filters out `gfx000` so CPU agents are not treated as GPU targets.
- Existing `--offload-arch`, `-offload-arch`, or `--amdgpu-target` flags suppress automatic injection to preserve user-provided target flags.
- If LOCAL probing cannot find a gfx target, the packager leaves compile flags unchanged so downstream build output remains the actionable failure point.

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

- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py` - 23 passed.

## Next Phase Readiness

Ready for Plan 02-03. The build template can now rely on staged `solution.json` carrying `compile_options.hip_cflags`.

---
*Phase: 02-rocm-schema-and-native-build-layer*
*Completed: 2026-05-21*
