---
phase: 79-docker-matrix-selection-and-preflight
plan: 1
subsystem: infra
tags: [docker, rocm, compatibility-matrix, preflight, pydantic]

requires:
  - phase: 78-matrix-contract-and-claim-guardrails
    provides: strict Matrix Target, Matrix Entry, claim boundary, and execution decision contracts
provides:
  - Checked-in declared ROCm Docker Target manifest for 7.0.x, 7.1.x, and 7.2.x
  - Pure Docker Target selection, build-arg construction, unknown override, and JSON preview helpers
  - CPU-safe Docker preflight classification for unsupported context and ROCm device access failures
affects: [phase-79-02, phase-80, phase-81, phase-82, docker-matrix-reports]

tech-stack:
  added: []
  patterns:
    - Strict frozen Pydantic v2 manifest and preflight models
    - Side-effect-free Docker Matrix selection and classification helpers
    - Shell-consumable JSON payloads for downstream script wiring

key-files:
  created:
    - docker/rocm-targets.json
    - src/sol_execbench/core/docker_matrix.py
    - tests/sol_execbench/test_docker_matrix_targets.py
    - tests/sol_execbench/test_docker_matrix_preflight.py
  modified: []

key-decisions:
  - "Declared Docker Target selection is repo-owned JSON policy, not runtime Docker tag discovery."
  - "Unknown Docker Targets require an explicit unsafe/untested override and remain `not_tested` with all authority fields false."
  - "Docker preflight failures use Phase 78 `runtime_unavailable` and `rocm_runtime_unavailable` semantics before benchmark execution."
  - "Selection and preflight JSON helpers do not execute Docker, pull images, probe ROCm hardware, or run benchmarks."

patterns-established:
  - "Docker Target manifests parse into strict entries and convert into Phase 78 `MatrixTarget` objects."
  - "Preflight observations preserve requested image repo/tag, nullable digest, build args, host device nodes, and GPU visibility environment."
  - "CLI JSON output exposes stable fields for `scripts/run_docker.sh` without live Docker requirements."

requirements-completed: [DOCKER-01, DOCKER-03, DOCKER-04, DOCKER-05]

duration: 6min
completed: 2026-05-28
---

# Phase 79 Plan 1: Docker Matrix Selection And Preflight Summary

**Declared ROCm Docker Targets with deterministic selection, non-authoritative unknown overrides, runtime-unavailable preflight classification, and shell-consumable Matrix JSON**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-28T06:23:41Z
- **Completed:** 2026-05-28T06:29:50Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added `docker/rocm-targets.json` with declared ROCm 7.0.2, 7.1.1 default, and 7.2.0 Docker Targets using `rocm/dev-ubuntu-24.04` complete tags.
- Added `src/sol_execbench/core/docker_matrix.py` with strict manifest parsing, default selection, MatrixTarget conversion, Docker build args, unknown override classification, preflight classification, and JSON CLI output.
- Added CPU-safe target and preflight tests covering default preservation, unknown rejection/override, nullable digest evidence, runtime-unavailable decisions, and JSON payload keys.

## Task Commits

1. **Task 1 RED: Docker target selection tests** - `27b4884` (test)
2. **Task 1 GREEN: Docker target manifest and selector** - `804cc91` (feat)
3. **Task 2 RED: Docker preflight tests** - `38352df` (test)
4. **Task 2 GREEN: Docker preflight classifier** - `bb2d2e1` (feat)
5. **Task 3: Shell-consumable Docker Matrix JSON** - `ccaa123` (feat)

## Files Created/Modified

- `docker/rocm-targets.json` - Checked-in declared Docker Target manifest with 7.0.x, default 7.1.x, and 7.2.x logical entries.
- `src/sol_execbench/core/docker_matrix.py` - Pure manifest loader, selector, MatrixTarget conversion, build args, unknown override, preflight classification, and module CLI.
- `tests/sol_execbench/test_docker_matrix_targets.py` - Manifest/default/selection/build-arg/override/preview JSON tests.
- `tests/sol_execbench/test_docker_matrix_preflight.py` - CPU-safe structured preflight classification and CLI JSON tests.

## Decisions Made

- Used JSON for the checked-in Docker Target manifest to keep the policy simple and shell-adjacent.
- Kept declared Target previews `not_tested` until a later live runtime validation phase supplies actual container or hardware evidence.
- Represented missing image digest as explicit `None`/JSON `null`; digest absence is diagnostic evidence, not a classification failure.
- Kept Docker execution out of this plan; Phase 79-02 will wire these helpers into `docker/Dockerfile` and `scripts/run_docker.sh`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

None. Optional `None` fields in `docker_matrix.py` represent not-yet-observed diagnostic evidence, explicit nullable digest evidence, or optional Target metadata. They do not block the plan goal.

## Threat Flags

None. The new manifest, Target selection, host observation, and diagnostic-to-benchmark trust boundaries were covered by the plan threat model and mitigated with strict models, unknown-target rejection, authority-false claim boundaries, and side-effect-free helpers.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_docker_matrix_targets.py -q` - passed, `11 passed in 1.20s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_docker_matrix_preflight.py -q` - passed, `7 passed in 1.16s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` - passed, `19 passed in 1.22s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` - passed, `47 passed in 1.21s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/docker_matrix.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py` - passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 79-02 can consume `python -m sol_execbench.core.docker_matrix preview` and `preflight` JSON to parameterize the Dockerfile and wrapper script while preserving the existing ROCm 7.1 default.

## Self-Check: PASSED

- Found created file: `docker/rocm-targets.json`.
- Found created file: `src/sol_execbench/core/docker_matrix.py`.
- Found created file: `tests/sol_execbench/test_docker_matrix_targets.py`.
- Found created file: `tests/sol_execbench/test_docker_matrix_preflight.py`.
- Found task commits: `27b4884`, `804cc91`, `38352df`, `bb2d2e1`, `ccaa123`.
- Re-ran plan verification successfully.

---
*Phase: 79-docker-matrix-selection-and-preflight*
*Completed: 2026-05-28*
