---
phase: 79-docker-matrix-selection-and-preflight
plan: 2
subsystem: infra
tags: [docker, rocm, compatibility-matrix, preflight, bash]

requires:
  - phase: 79-docker-matrix-selection-and-preflight
    provides: Docker Target manifest, selector, preflight classifier, and JSON helper from Plan 79-01
provides:
  - Default-preserving parameterized ROCm Dockerfile base image selection
  - Docker wrapper Target selection flags and ROCm build-arg wiring
  - Matrix-compatible Docker preflight diagnostics before Docker build/run or benchmark execution
affects: [phase-80, phase-81, phase-82, docker-workflow, compatibility-matrix]

tech-stack:
  added: []
  patterns:
    - Shell wrapper delegates Docker Target policy and preflight classification to Python Matrix helpers
    - CPU-safe script tests use env-injected observations and dry-run command previews

key-files:
  created:
    - tests/sol_execbench/test_run_docker_matrix_script.py
  modified:
    - docker/Dockerfile
    - scripts/run_docker.sh

key-decisions:
  - "The Dockerfile keeps `rocm/dev-ubuntu-24.04:7.1.1-complete` as the default through pre-FROM build args."
  - "The Docker wrapper resolves Target selection before host preflight so unknown Targets fail before Docker build/run."
  - "Script tests use `SOL_EXECBENCH_RUN_DOCKER_DRY_RUN=1` and env-injected observations to avoid live Docker or ROCm hardware."
  - "Runtime-unavailable and preflight-only paths emit the helper JSON directly instead of shell-only error text."

patterns-established:
  - "Use `--target`, `--allow-unknown-target`, and `--preflight-only` as wrapper-owned flags before the `--` command separator."
  - "Pass `ROCM_DOCKER_IMAGE` and `ROCM_DOCKER_TAG` to `docker build` from the declared Target helper payload."
  - "Keep Docker run permissions targeted to `/dev/kfd`, `/dev/dri`, video group, seccomp unconfined, and ipc host; do not add `--privileged`."

requirements-completed: [DOCKER-02, DOCKER-03, DOCKER-04, DOCKER-05]

duration: 8min
completed: 2026-05-28
---

# Phase 79 Plan 2: Docker Matrix Selection And Preflight Summary

**Dockerfile and wrapper integration for declared ROCm Target selection, selected build args, and runtime-unavailable preflight stops**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-28T06:33:18Z
- **Completed:** 2026-05-28T06:41:39Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Parameterized `docker/Dockerfile` with pre-`FROM` `ROCM_DOCKER_IMAGE` and `ROCM_DOCKER_TAG` args while preserving the existing ROCm 7.1.1 default image.
- Extended `scripts/run_docker.sh` with `--target`, `--allow-unknown-target`, `--preflight-only`, manifest-backed build args, and CPU-safe dry-run command previews.
- Replaced generic host preflight shell exits with Matrix-compatible JSON diagnostics for Docker Desktop, missing device nodes, and GPU access failures before Docker build/run.

## Task Commits

1. **Task 1 RED: Dockerfile base arg tests** - `43c79c1` (test)
2. **Task 1 GREEN: Parameterized Dockerfile base image** - `042df1f` (feat)
3. **Task 2 RED: run_docker Target tests** - `8e843aa` (test)
4. **Task 2 GREEN: Target parser and build args** - `dc528fe` (feat)
5. **Task 3 RED: preflight script tests** - `bbd5339` (test)
6. **Task 3 GREEN: Matrix preflight wrapper classification** - `a1e6fb3` (feat)

## Files Created/Modified

- `docker/Dockerfile` - Uses default-preserving ROCm base image/tag args before the first `FROM`.
- `scripts/run_docker.sh` - Resolves Docker Target selection, passes selected ROCm build args, emits preflight diagnostics, and preserves targeted ROCm device run flags.
- `tests/sol_execbench/test_run_docker_matrix_script.py` - CPU-safe Dockerfile/static and wrapper subprocess tests for default selection, Target build args, unknown rejection, and preflight stops.

## Decisions Made

- Used Python helper JSON from Plan 79-01 as the shell contract rather than duplicating manifest policy in Bash.
- Added `SOL_EXECBENCH_RUN_DOCKER_DRY_RUN=1` only as a test/preview path; normal script execution still runs Docker when preflight allows it.
- Env-injected preflight observations are accepted for tests, while normal execution collects `docker context`, Docker host, `/dev/kfd`, and `/dev/dri` state directly.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

None. The `DOCKER_TARGET=""` shell initialization is parser state, not placeholder data.

## Threat Flags

None. The plan's threat model covered CLI Target parsing, helper JSON to shell build args, host preflight state, diagnostic spoofing boundaries, and Docker device exposure. The wrapper continues using targeted ROCm device flags and does not add `--privileged`.

## Verification

- `bash -n scripts/run_docker.sh` - passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_docker_matrix_script.py -q` - passed, `10 passed in 1.38s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_docker_matrix_script.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` - passed, `29 passed in 2.01s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py -q` - passed, `57 passed in 1.98s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/docker_matrix.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py` - passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 80 can now rely on Docker Target selection and selected ROCm image/tag build args being available through the standard wrapper. Phase 81 can build on the preflight JSON path for richer runtime evidence and aggregate compatibility reports.

## Self-Check: PASSED

- Found created file: `tests/sol_execbench/test_run_docker_matrix_script.py`.
- Found modified file: `docker/Dockerfile`.
- Found modified file: `scripts/run_docker.sh`.
- Found task commits: `43c79c1`, `042df1f`, `8e843aa`, `dc528fe`, `bbd5339`, `a1e6fb3`.
- Re-ran plan verification successfully.

---
*Phase: 79-docker-matrix-selection-and-preflight*
*Completed: 2026-05-28*
