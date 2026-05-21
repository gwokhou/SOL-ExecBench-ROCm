---
phase: 01-rocm-environment-baseline
plan: 01
subsystem: infra
tags: [docker, rocm, hip, container]
requires: []
provides:
  - Pinned ROCm 7.1.1 Docker base image
  - AMD ROCm Docker runtime device passthrough
  - ROCm-aware container entrypoint startup checks
affects: [docker, runtime, phase-2, phase-3]
tech-stack:
  added: [rocm/dev-ubuntu-24.04:7.1.1-complete]
  patterns: [best-effort-rocm-startup, targeted-device-passthrough]
key-files:
  created: []
  modified:
    - docker/Dockerfile
    - scripts/run_docker.sh
    - docker/entrypoint.sh
key-decisions:
  - "Use rocm/dev-ubuntu-24.04:7.1.1-complete as the reproducible ROCm baseline."
  - "Use targeted /dev/kfd and /dev/dri passthrough instead of NVIDIA runtime flags."
  - "Keep clock locking best-effort until ROCm timing work in Phase 3."
patterns-established:
  - "ROCm PyTorch still uses torch.cuda for device visibility while torch.version.hip proves the HIP backend."
requirements-completed: [ENV-01, ENV-02]
duration: 35min
completed: 2026-05-21
---

# Phase 1 Plan 01: ROCm Container Baseline Summary

**Pinned ROCm Docker image, AMD GPU passthrough flags, and ROCm-aware entrypoint startup behavior**

## Performance

- **Duration:** 35 min
- **Started:** 2026-05-21T05:00:00Z
- **Completed:** 2026-05-21T05:36:48Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Replaced the NVIDIA CUDA base image with `rocm/dev-ubuntu-24.04:7.1.1-complete`.
- Switched Docker launch defaults from `--gpus all` / privileged mode to ROCm device passthrough.
- Updated the entrypoint to detect `torch.version.hip`, warn on missing ROCm GPU visibility, and only unlock clocks when a lock was acquired.

## Task Commits

1. **Task 1-3: ROCm Docker baseline and entrypoint** - `aa4d075` (feat)

## Files Created/Modified

- `docker/Dockerfile` - ROCm base image, ROCm environment variables, ROCm tool smoke checks, and ROCm SMI sudo handling.
- `scripts/run_docker.sh` - AMD device passthrough flags and ROCm-oriented usage example.
- `docker/entrypoint.sh` - HIP backend probe, best-effort clock locking, non-blocking trace directory warning.

## Decisions Made

The Docker runtime uses targeted AMD device flags rather than broad privileged mode. This keeps the Phase 1 environment baseline narrow and leaves deeper timing/clock behavior to Phase 3.

## Deviations from Plan

None. The final entrypoint uses the literal `torch.version.hip` expression expected by the plan verification.

## Issues Encountered

Docker build validation showed the ROCm base image may already contain UID 1000. The Dockerfile now reuses existing UID/GID users instead of unconditionally creating a new user. Host-level runtime validation also showed `/dev/kfd` is absent in the current shell environment; that is recorded in Plan 03.

## User Setup Required

None.

## Next Phase Readiness

Phase 2 can rely on the Docker image and run wrapper being ROCm-first. Phase 3 still needs to implement ROCm-specific timing and clock management semantics.

## Verification

- `grep -v '^#' docker/Dockerfile | grep -q 'rocm/dev-ubuntu-24.04:7.1.1-complete' && ! grep -v '^#' docker/Dockerfile | grep -Eq 'nvidia/cuda|CUDA_HOME|CUDACXX|nvcc|CUTLASS_DIR|/usr/local/cuda'` - PASS
- `grep -q -- '--device=/dev/kfd' scripts/run_docker.sh && grep -q -- '--device=/dev/dri' scripts/run_docker.sh && grep -q -- '--group-add video' scripts/run_docker.sh && ! grep -v '^#' scripts/run_docker.sh | grep -Eq -- '--gpus|--privileged'` - PASS
- `grep -q 'torch.version.hip' docker/entrypoint.sh && ! grep -v '^#' docker/entrypoint.sh | grep -q 'No CUDA device detected'` - PASS
- `bash -n docker/entrypoint.sh` - PASS
- `bash -n scripts/run_docker.sh` - PASS

## Self-Check: PASSED

---
*Phase: 01-rocm-environment-baseline*
*Completed: 2026-05-21*
