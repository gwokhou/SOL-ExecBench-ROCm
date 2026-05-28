---
phase: 79-docker-matrix-selection-and-preflight
verified: 2026-05-28T06:57:16Z
status: passed
score: 8/8 must-haves verified
overrides_applied: 0
---

# Phase 79: Docker Matrix Selection And Preflight Verification Report

**Phase Goal:** Users can select declared ROCm Docker Targets and get conservative preflight classification before benchmark execution.
**Verified:** 2026-05-28T06:57:16Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can select checked-in ROCm Docker Targets for configured 7.0.x, 7.1.x, and 7.2.x entries while the existing ROCm 7.1 default path still works. | VERIFIED | `docker/rocm-targets.json` declares `7.0.2-complete`, default `7.1.1-complete`, and `7.2.0-complete`; `select_docker_target(None)` uses the manifest default; dry-run `./scripts/run_docker.sh --build` emits 7.1 build args and Docker run command. |
| 2 | User can see exact requested image repository/tag, resolved digest when available, and Docker build arguments in compatibility evidence. | VERIFIED | `DockerPreflightObservation` stores repo/tag/digest/build args; preview payload includes `image_repository`, `image_tag`, nullable `image_digest`, and `build_args`; default preview printed `rocm/dev-ubuntu-24.04`, `7.1.1-complete`, and digest `null`. |
| 3 | User is blocked from unknown ROCm Targets unless they explicitly choose an unsafe or untested override. | VERIFIED | `select_docker_target` raises for unknown ids without override; override creates `unsafe-untested-*`, `not_tested`, `target_not_tested`, and `benchmark_allowed=false`; `run_docker.sh` exits before build/run for unsafe override target ids. |
| 4 | User receives `runtime_unavailable` before benchmark execution when Docker context, `/dev/kfd`, `/dev/dri`, or GPU device access is unavailable. | VERIFIED | `_runtime_unavailable_reason` covers Docker Desktop, missing/inaccessible `/dev/kfd`, missing/inaccessible `/dev/dri`, and `gpu_accessible=false`; `run_docker.sh` prints preflight JSON and exits before build/run. |
| 5 | Dockerfile preserves the existing ROCm 7.1 base image default while accepting selected image/tag build args. | VERIFIED | `docker/Dockerfile` has pre-`FROM` args `ROCM_DOCKER_IMAGE=rocm/dev-ubuntu-24.04` and `ROCM_DOCKER_TAG=7.1.1-complete`, then `FROM ${ROCM_DOCKER_IMAGE}:${ROCM_DOCKER_TAG} AS base`. |
| 6 | `scripts/run_docker.sh` wires Target flags, Python helper JSON, Docker build args, and targeted ROCm run args. | VERIFIED | Script parses `--target`, `--allow-unknown-target`, `--preflight-only`; calls `python -m sol_execbench.core.docker_matrix`; passes `ROCM_DOCKER_IMAGE`/`ROCM_DOCKER_TAG`; keeps `/dev/kfd`, `/dev/dri`, video group, seccomp, and ipc flags without `--privileged`. |
| 7 | Not-tested and preflight paths remain diagnostic/non-authoritative. | VERIFIED | Preview and preflight payloads include `benchmark_allowed=false` for `not_tested` and `runtime_unavailable`, with score, paper-parity, leaderboard, container, and native-host authority fields false. |
| 8 | Focused CPU-safe tests prove manifest, selection, preflight, script, and Dockerfile behavior. | VERIFIED | `tests/sol_execbench/test_docker_matrix_targets.py`, `test_docker_matrix_preflight.py`, and `test_run_docker_matrix_script.py` cover the scoped behavior; focused 32-test suite and 60-test Phase 79 guardrail suite passed. |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docker/rocm-targets.json` | Checked-in declared Docker Target manifest | VERIFIED | Exists, substantive, default id points to 7.1.1, includes 7.0.2 and 7.2.0 container user-space entries. |
| `src/sol_execbench/core/docker_matrix.py` | Pure selection, build-arg, override, digest, and preflight helpers | VERIFIED | Exists, substantive, imports Phase 78 Matrix contracts, exposes preview/preflight CLI, used by script and tests. |
| `docker/Dockerfile` | Parameterized ROCm base image selection | VERIFIED | Pre-`FROM` image/tag args preserve 7.1 default and feed `FROM`. |
| `scripts/run_docker.sh` | Target selection, preflight-only mode, build-arg wiring, runtime-unavailable stop | VERIFIED | Calls Python helper, parses flags before `--`, emits diagnostics, gates build/run, preserves targeted device flags. |
| `tests/sol_execbench/test_docker_matrix_targets.py` | Manifest/default/selection/override/preview tests | VERIFIED | Covers declared entries, default, build args, unknown rejection, override non-authority, JSON preview. |
| `tests/sol_execbench/test_docker_matrix_preflight.py` | Runtime-unavailable and evidence tests | VERIFIED | Covers Docker Desktop, `/dev/kfd`, `/dev/dri`, GPU access, nullable digest, authority flags, invalid booleans. |
| `tests/sol_execbench/test_run_docker_matrix_script.py` | Dockerfile/script static and subprocess tests | VERIFIED | Covers Dockerfile args, script flags, dry-run build args, unknown rejection, preflight blocking, no `--privileged`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `docker/rocm-targets.json` | `src/sol_execbench/core/docker_matrix.py` | Manifest loader | VERIFIED | `load_docker_target_manifest` reads and validates the checked-in JSON into strict models. |
| `src/sol_execbench/core/docker_matrix.py` | `src/sol_execbench/core/compatibility.py` | Phase 78 Matrix imports | VERIFIED | Imports `MatrixTarget`, `MatrixObservedEvidence`, statuses, reason codes, claim boundary, builder, and classifier. |
| `tests/sol_execbench/test_docker_matrix_preflight.py` | `src/sol_execbench/core/docker_matrix.py` | Structured observations | VERIFIED | Tests construct `DockerPreflightObservation` and assert runtime-unavailable decisions. |
| `scripts/run_docker.sh` | `src/sol_execbench/core/docker_matrix.py` | Python module JSON | VERIFIED | Script invokes `python -m sol_execbench.core.docker_matrix preview` and `preflight`. |
| `scripts/run_docker.sh` | `docker/Dockerfile` | Docker build args | VERIFIED | Script passes `--build-arg ROCM_DOCKER_IMAGE=...` and `--build-arg ROCM_DOCKER_TAG=...`. |
| `tests/sol_execbench/test_run_docker_matrix_script.py` | `scripts/run_docker.sh` | Static/subprocess checks | VERIFIED | Tests execute dry-run and env-injected preflight paths. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `docker_matrix.py` | Docker Target selection | `docker/rocm-targets.json` via `load_docker_target_manifest` | Yes - strict parsed manifest entries | VERIFIED |
| `docker_matrix.py` | Build args | `docker_build_args_for_target(selection.target)` | Yes - repository/tag from selected Target | VERIFIED |
| `docker_matrix.py` | Preflight evidence | `DockerPreflightObservation` fields | Yes - explicit shell-collected or test-injected observations | VERIFIED |
| `scripts/run_docker.sh` | Selected image/tag | Helper JSON `build_args.*` parsed by `matrix_json_value` | Yes - used in `docker build` command | VERIFIED |
| `scripts/run_docker.sh` | Runtime gate | Helper preflight JSON `status` and `benchmark_allowed` | Yes - controls exit before build/run | VERIFIED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Shell syntax | `bash -n scripts/run_docker.sh` | Exit 0 | PASS |
| Focused Phase 79 tests | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py -q` | `32 passed in 1.39s` | PASS |
| Phase 79 guardrail suite | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py -q` | `60 passed in 1.40s` | PASS |
| Ruff | `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/docker_matrix.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py` | `All checks passed!` | PASS |
| Default preview | `PYTHONPATH=src python -m sol_execbench.core.docker_matrix preview --manifest docker/rocm-targets.json` | JSON with default 7.1.1 image/tag, digest `null`, status `not_tested`, authority false | PASS |
| Unknown target rejection | `PYTHONPATH=src python -m sol_execbench.core.docker_matrix preview --manifest docker/rocm-targets.json --target rocm-9.9-unknown` | Exit 1 with `Unknown Docker Target` before any Docker operation | PASS |
| Default script dry-run | `SOL_EXECBENCH_RUN_DOCKER_DRY_RUN=1 PYTHONPATH=src ./scripts/run_docker.sh --build` | Emits 7.1.1 build args and targeted Docker run command | PASS |
| Runtime-unavailable script preflight | env-injected Docker Desktop/GPU inaccessible `./scripts/run_docker.sh --preflight-only` | Exit 1 with `runtime_unavailable`, `rocm_runtime_unavailable`, repo/tag/digest/build args | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| Conventional probes | `find scripts -path '*/tests/probe-*.sh' -type f` | No probes found | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DOCKER-01 | 79-01 | Checked-in ROCm Docker matrix manifest for configured 7.0.x, 7.1.x, 7.2.x Targets | SATISFIED | Manifest contains 7.0.2, default 7.1.1, and 7.2.0 complete entries. |
| DOCKER-02 | 79-02 | Dockerfile supports parameterized ROCm base image while preserving 7.1 default | SATISFIED | Pre-`FROM` args and interpolated `FROM` line preserve existing base image. |
| DOCKER-03 | 79-01, 79-02 | Script supports declared Target selection and rejects unknown Targets unless explicitly unsafe/untested | SATISFIED | Python selector rejects unknown ids; script parses Target flags and blocks unsafe override before Docker commands. |
| DOCKER-04 | 79-01, 79-02 | Preflight classifies missing Docker/device/GPU runtime as `runtime_unavailable` before benchmark execution | SATISFIED | Python classifier and script gate cover Docker Desktop, `/dev/kfd`, `/dev/dri`, and GPU access. |
| DOCKER-05 | 79-01, 79-02 | Docker reports record requested repo/tag, digest when available, and build arguments | SATISFIED | Preview/preflight payloads expose repo/tag, nullable digest, and `ROCM_DOCKER_IMAGE`/`ROCM_DOCKER_TAG`. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | No TODO/FIXME/XXX/HACK/placeholder/stub markers found in scoped files. |

### Human Verification Required

None for Phase 79's CPU-safe contract. Live Docker/ROCm container execution and richer runtime evidence are explicitly deferred to later phases.

### Gaps Summary

No blocking gaps found. The phase delivers declared Docker Target selection, default 7.1 preservation, unknown/not-tested conservative blocking, runtime-unavailable preflight classification, Matrix-compatible evidence fields, Docker build/run wiring, and focused tests.

---

_Verified: 2026-05-28T06:57:16Z_
_Verifier: the agent (gsd-verifier)_
