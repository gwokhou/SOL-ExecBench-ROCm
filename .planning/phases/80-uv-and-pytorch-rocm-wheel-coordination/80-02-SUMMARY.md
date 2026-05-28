---
phase: 80-uv-and-pytorch-rocm-wheel-coordination
plan: 2
subsystem: infra
tags: [docker, dependency-preflight, pytorch, rocm, uv, pytest]
requires:
  - phase: 80-uv-and-pytorch-rocm-wheel-coordination
    plan: 1
    provides: PyTorch ROCm dependency policy, classifier, and shell-consumable preflight JSON.
  - phase: 79-docker-matrix-selection-and-preflight
    provides: Declared Docker Target selection and Docker runtime preflight wrapper pattern.
provides:
  - Docker wrapper dependency preflight gating before Docker build/run command construction.
  - CPU-safe wrapper tests for mixed-version and unavailable PyTorch ROCm dependency states.
  - Explicit mixed-version dependency debug override separate from unknown Target override.
affects: [phase-81-runtime-evidence, phase-82-validation-docs]
tech-stack:
  added: []
  patterns: [thin-shell-json-gate, env-injected-script-observations, explicit-debug-override]
key-files:
  created:
    - tests/sol_execbench/test_run_docker_dependency_preflight.py
  modified:
    - scripts/run_docker.sh
    - src/sol_execbench/core/dependency_matrix.py
key-decisions:
  - "The Docker wrapper delegates dependency policy to `python -m sol_execbench.core.dependency_matrix preflight` and gates only on helper JSON fields."
  - "Injected wrapper observations use `SOL_EXECBENCH_DEPENDENCY_*` names so script tests stay CPU-safe and avoid live Docker, ROCm hardware, uv lock mutation, or wheel installation."
  - "`--allow-mixed-version-dependencies` and `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES=1` are dependency-specific debug toggles and do not reuse `--allow-unknown-target`."
requirements-completed: [DEPS-06, DEPS-07]
duration: 5min
completed: 2026-05-28
---

# Phase 80 Plan 2: Docker Wrapper Dependency Preflight Summary

**Docker wrapper dependency preflight blocks illegal PyTorch ROCm stacks before build/run while preserving bounded debug semantics**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-28T09:31:01Z
- **Completed:** 2026-05-28T09:36:08Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `tests/sol_execbench/test_run_docker_dependency_preflight.py` with CPU-safe subprocess coverage for mixed-version, unavailable-wheel, matching-default, unknown-target override, and dependency debug override behavior.
- Updated `scripts/run_docker.sh` to invoke dependency preflight after declared Docker Target selection and before Docker build/run or benchmark command text.
- Added `SOL_EXECBENCH_DEPENDENCY_*` observation pass-through for wrapper tests and debugging without live Docker, ROCm hardware, uv lock mutation, or wheel installation.
- Added `--allow-mixed-version-dependencies` and `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES=1`, mapped to helper `--allow-mixed-version-debug`, while keeping benchmark, validation, score, paper-parity, and leaderboard authority false.
- Fixed the dependency helper CLI integration path so calls without injected observations use the existing live dependency collector instead of classifying an all-null observation.

## Task Commits

1. **Task 1 RED: Dependency preflight wrapper gate tests** - `d7e4a94`
2. **Task 1 GREEN: Dependency preflight wrapper gate** - `622bc2f`
3. **Task 2 RED: Mixed-version dependency override tests** - `2208b38`
4. **Task 2 GREEN: Mixed-version dependency debug override** - `5f5a14c`

## Files Created/Modified

- `scripts/run_docker.sh` - Added dependency preflight JSON invocation, env-injected observation arguments, default blocking, and the explicit mixed-version dependency debug override.
- `tests/sol_execbench/test_run_docker_dependency_preflight.py` - Added CPU-safe wrapper subprocess tests for DEPS-06 and DEPS-07.
- `src/sol_execbench/core/dependency_matrix.py` - Narrow integration fix so uninjected CLI calls collect live dependency observations.

## Decisions Made

- Kept Bash policy thin: the wrapper checks `status` and `benchmark_allowed` from helper JSON instead of duplicating dependency policy.
- Kept preflight-only runtime tests compatible by using dependency JSON as the preflight-only output only when dependency observation overrides are present.
- Kept the debug override bounded: the wrapper still exits before normal Docker build/run when `benchmark_allowed=false`, even if probe/smoke fields are true.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking integration issue] Fixed uninjected helper CLI observation handling**
- **Found during:** Task 1
- **Issue:** `dependency_matrix.py preflight` had a live observation collector, but the CLI path built an all-null observation when no test overrides were passed. That would make normal wrapper integration classify live environments as unavailable.
- **Fix:** Added `_observation_args_present()` and `_observation_from_args()` so the CLI uses injected values when present and otherwise calls `collect_pytorch_dependency_observation()`.
- **Files modified:** `src/sol_execbench/core/dependency_matrix.py`
- **Commit:** `622bc2f`

## Verification

- `bash -n scripts/run_docker.sh` - passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_docker_dependency_preflight.py -q` - 7 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_dependency_matrix_cli.py tests/sol_execbench/test_run_docker_dependency_preflight.py -q` - 89 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/dependency_matrix.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_dependency_matrix_cli.py tests/sol_execbench/test_run_docker_dependency_preflight.py` - passed
- `git status --short pyproject.toml uv.lock` - no modifications

## Known Stubs

None.

## Threat Flags

None.

## User Setup Required

None.

## Next Phase Readiness

Phase 81 can build runtime evidence/reporting on top of a wrapper that now blocks illegal dependency stacks before Docker build/run and preserves non-authoritative debug diagnostics for mixed-version states.

## Self-Check: PASSED

- Summary file exists.
- Task commits exist: `d7e4a94`, `622bc2f`, `2208b38`, `5f5a14c`.
- Created file exists: `tests/sol_execbench/test_run_docker_dependency_preflight.py`.
- Modified files exist: `scripts/run_docker.sh`, `src/sol_execbench/core/dependency_matrix.py`.

---
*Phase: 80-uv-and-pytorch-rocm-wheel-coordination*
*Completed: 2026-05-28*
