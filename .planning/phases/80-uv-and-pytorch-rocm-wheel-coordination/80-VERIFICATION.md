---
phase: 80-uv-and-pytorch-rocm-wheel-coordination
verified: 2026-05-28T09:48:29Z
status: passed
score: 12/12 must-haves verified
overrides_applied: 0
---

# Phase 80: uv And PyTorch ROCm Wheel Coordination Verification Report

**Phase Goal:** Users can tell whether a selected ROCm Target has a matching PyTorch ROCm dependency stack before any clean validation claim is made.
**Verified:** 2026-05-28T09:48:29Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | User can inspect each Target's PyTorch ROCm wheel/index policy, including expected local-version tag and uv index or lock strategy. | VERIFIED | `docker/rocm-targets.json` declares `pytorch_dependency_policy` for ROCm 7.0, 7.1, and 7.2 Targets with policy id, expected local version, uv index name/url, lock strategy, suggested uv command, and Triton policy. |
| 2 | User can keep using the default ROCm 7.1 dependency path unless a per-Target workflow is explicitly selected and recorded. | VERIFIED | Default Target remains `rocm-7.1.1-ubuntu-24.04-container`; `pyproject.toml` and `uv.lock` retain `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, `triton-rocm==3.6.0`, `pytorch-rocm71`, and `pytorch-rocm-root`. |
| 3 | Missing or unsupported PyTorch ROCm wheels classify as `pytorch_wheel_unavailable`, not benchmark failures. | VERIFIED | `dependency_matrix.py` returns `PYTORCH_WHEEL_UNAVAILABLE` for unavailable policy, import error, or missing torch distribution, with reason code `pytorch_rocm_wheel_unavailable`; tests assert benchmark and authority are denied. |
| 4 | CPU, CUDA, wrong-index, wrong-ROCm, Triton, or toolchain mismatches classify as `mixed_version`. | VERIFIED | `_mismatch_reason` covers CUDA runtime, wrong torch version/local tag/ROCm target/HIP prefix, torchvision, Triton status/version, container ROCm, and toolchain ROCm mismatches; parametrized tests assert `mixed_version` and `target_observed_mismatch`. |
| 5 | Illegal mixed-version validation is blocked by default, while explicit debug override can continue probes/smoke without clean validation or authority. | VERIFIED | `classify_matrix_entry_for_execution` is called for every dependency preflight; default mixed-version decisions deny benchmark/probes/smoke, while debug override permits probes/smoke only and keeps validation/authority false. |
| 6 | Matrix Entry JSON records dependency policy, not just helper text. | VERIFIED | `MatrixDependencyPolicyEvidence` is a strict model under `MatrixObservedEvidence.dependency_policy`; `DependencyPreflightResult.to_preview_payload()` derives shell JSON from `entry.model_dump(mode="json")`. |
| 7 | Shell-consumable dependency preflight JSON exposes policy, status, execution decision, and authority fields. | VERIFIED | `python -m sol_execbench.core.dependency_matrix preflight` emits sorted JSON with target, policy, expected versions, status/reason, benchmark/probe/smoke, validation, and authority fields; CLI tests compare JSON fields to the Matrix Entry policy payload. |
| 8 | Docker wrapper runs dependency preflight after Target selection and before Docker build/run or benchmark command construction. | VERIFIED | `scripts/run_docker.sh` resolves Target JSON first, then invokes `sol_execbench.core.dependency_matrix preflight` and exits on blocked dependency status before `docker build`/`docker run` command construction. |
| 9 | Unknown Target override remains separate from mixed-version dependency debug override. | VERIFIED | `--allow-unknown-target` and `--allow-mixed-version-dependencies` set separate parser variables; tests assert unknown Target override alone does not enable dependency debug semantics. |
| 10 | CPU-safe tests prove dependency gating without live Docker, ROCm hardware, uv lock mutation, or PyTorch wheel installation. | VERIFIED | Tests inject observations via `SOL_EXECBENCH_DEPENDENCY_*` env vars and dry-run/preflight-only subprocesses; no probe requires live Docker or ROCm hardware. |
| 11 | DEPS-01 through DEPS-07 are covered by source and tests. | VERIFIED | Requirements coverage table below maps every requirement to implementation evidence and passing tests. |
| 12 | Code review found no source-level defects in the phase files. | VERIFIED | `80-REVIEW.md` reports status `clean`, 9 files reviewed, 0 critical/warning/info findings; prior CR-01 was closed by test coverage for torch import errors. |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `docker/rocm-targets.json` | Target-adjacent dependency policy | VERIFIED | Exists, substantive, parsed by strict policy loader. |
| `src/sol_execbench/core/compatibility.py` | Strict Matrix Entry dependency policy evidence | VERIFIED | Defines `MatrixDependencyPolicyEvidence` and `MatrixObservedEvidence.dependency_policy`. |
| `src/sol_execbench/core/docker_matrix.py` | Docker Target carries policy | VERIFIED | `DockerTargetManifestEntry` accepts `pytorch_dependency_policy`. |
| `src/sol_execbench/core/dependency_matrix.py` | Policy loader, observation collector, classifier, JSON CLI | VERIFIED | Strict models, classifier, Matrix Entry construction, execution decision, and module CLI are present. |
| `scripts/run_docker.sh` | Wrapper dependency preflight gate | VERIFIED | Calls dependency helper, passes env observations, blocks before Docker commands, has explicit dependency debug override. |
| `tests/sol_execbench/test_dependency_matrix_policy.py` | Policy/default preservation tests | VERIFIED | Covers manifest policy, default ROCm 7.1 path, explicit ROCm 7.0/7.2 workflows, Matrix Entry policy payload. |
| `tests/sol_execbench/test_dependency_matrix_classification.py` | Classification tests | VERIFIED | Covers unavailable wheels/import errors, mixed-version mismatches, matching not-tested, debug override authority denial. |
| `tests/sol_execbench/test_dependency_matrix_cli.py` | CLI JSON tests | VERIFIED | Covers default JSON, payload parity with Matrix Entry, ROCm 7.2 mismatch, debug override, invalid boolean error. |
| `tests/sol_execbench/test_run_docker_dependency_preflight.py` | Wrapper gate tests | VERIFIED | Covers mixed/unavailable blocking, matching default not-tested gate, override separation, dependency debug override semantics. |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `docker/rocm-targets.json` | `src/sol_execbench/core/dependency_matrix.py` | `pytorch_dependency_policy` parsed into strict `PytorchDependencyPolicy` | VERIFIED | SDK key-link check passed. |
| `src/sol_execbench/core/dependency_matrix.py` | `src/sol_execbench/core/compatibility.py` | Policy evidence, Python/toolchain evidence, Matrix statuses, and execution decisions | VERIFIED | SDK key-link check passed; code imports and constructs Matrix evidence. |
| `tests/sol_execbench/test_dependency_matrix_policy.py` | `src/sol_execbench/core/compatibility.py` | Builds `MatrixObservedEvidence(dependency_policy=...)` and asserts `entry.model_dump(mode="json")` policy payload | VERIFIED | Manually verified. SDK check false-negative: target path not referenced literally, but imported types are used. |
| `src/sol_execbench/core/dependency_matrix.py` | `src/sol_execbench/core/docker_matrix.py` | Target selection and Target-to-Matrix conversion | VERIFIED | SDK key-link check passed. |
| `tests/sol_execbench/test_dependency_matrix_policy.py` | `pyproject.toml` | Static assertions preserve ROCm 7.1 default dependency path | VERIFIED | Manually verified assertions for torch, torchvision, triton-rocm, indexes, and uv lock URLs. |
| `scripts/run_docker.sh` | `src/sol_execbench/core/dependency_matrix.py` | Module CLI `preflight` and JSON field consumption | VERIFIED | SDK key-link check passed. |
| `scripts/run_docker.sh` | `src/sol_execbench/core/docker_matrix.py` | Dependency preflight runs after declared Target selection | VERIFIED | SDK key-link check passed. |
| `tests/sol_execbench/test_run_docker_dependency_preflight.py` | `scripts/run_docker.sh` | Dry-run subprocess with env-injected observations | VERIFIED | SDK key-link check passed. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| `dependency_matrix.py` | `policy_payload` | `entry.model_dump(mode="json")["observed"]["dependency_policy"]` built from selected Target policy | Yes | VERIFIED |
| `dependency_matrix.py` | `status`, `reason_code`, `decision` | `_classify_observation()` plus `classify_matrix_entry_for_execution()` | Yes | VERIFIED |
| `scripts/run_docker.sh` | `DEPENDENCY_PREFLIGHT_JSON` | `python -m sol_execbench.core.dependency_matrix preflight` | Yes | VERIFIED |
| `scripts/run_docker.sh` | gate decision | JSON `status` and `benchmark_allowed` fields | Yes | VERIFIED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Shell syntax valid | `bash -n scripts/run_docker.sh` | exit 0 | PASS |
| Matrix/dependency regression suite | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_dependency_matrix_cli.py tests/sol_execbench/test_run_docker_dependency_preflight.py -q` | `90 passed in 2.49s` | PASS |
| Ruff on phase files | `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/dependency_matrix.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_dependency_matrix_cli.py tests/sol_execbench/test_run_docker_dependency_preflight.py` | `All checks passed!` | PASS |
| Code review | Read `.planning/phases/80-uv-and-pytorch-rocm-wheel-coordination/80-REVIEW.md` | clean; 0 findings across 9 files | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| n/a | Probe discovery found no `scripts/*/tests/probe-*.sh` and no phase-declared probes. | No probes required for this phase. | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| DEPS-01 | 80-01 | Each Matrix Entry records PyTorch ROCm wheel/index policy, including expected local-version tag and uv index or lock strategy. | SATISFIED | Strict policy evidence in `compatibility.py`; manifest policy converted into `observed.dependency_policy`; tests assert `entry.model_dump(mode="json")` payload. |
| DEPS-02 | 80-01, 80-02 | Default project dependency path remains ROCm 7.1 unless explicit per-Target workflow selected and recorded. | SATISFIED | Default manifest policy is ROCm 7.1 project default; pyproject/uv lock retain ROCm 7.1 pins and indexes; ROCm 7.0/7.2 policies are explicit workflows. |
| DEPS-03 | 80-01, 80-02 | Missing or unsupported PyTorch ROCm wheels classify as `pytorch_wheel_unavailable`, not benchmark failures. | SATISFIED | Classifier handles unavailable policy, torch import error, and missing torch distribution; tests assert status/reason and wrapper blocks before Docker command text. |
| DEPS-04 | 80-01 | CPU, CUDA, wrong-index, or wrong-ROCm PyTorch wheels are detected from metadata and runtime probes. | SATISFIED | Observation model captures distribution/runtime/local tag/HIP/CUDA/device data; mismatch logic catches CPU/no local tag, CUDA, wrong local tag, wrong target, wrong HIP. |
| DEPS-05 | 80-01, 80-02 | PyTorch wheel, container ROCm user-space, Triton ROCm, or toolchain mismatch classifies as `mixed_version`. | SATISFIED | `_mismatch_reason` covers torch, torchvision, Triton, container, and toolchain mismatches; tests assert `mixed_version`/`target_observed_mismatch`. |
| DEPS-06 | 80-02 | Illegal `mixed_version` Targets are blocked during preflight before benchmark execution by default. | SATISFIED | `run_docker.sh` exits on `mixed_version`, `pytorch_wheel_unavailable`, or `benchmark_allowed != True` before build/run; tests assert no Docker command text. |
| DEPS-07 | 80-01, 80-02 | Explicit debug override may allow probes/smoke only, never validation or authority claims. | SATISFIED | `--allow-mixed-version-dependencies` maps to helper `--allow-mixed-version-debug`; classifier and wrapper tests assert probes/smoke true but benchmark/validation/authority false. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---:|---|---|---|
| n/a | n/a | No `TBD`, `FIXME`, `XXX`, `TODO`, placeholder, empty implementation, or console-only implementation matches in phase files. | n/a | None |

### Human Verification Required

None. Phase 80 behaviors are CPU-safe policy/classifier/CLI/shell-gate behaviors and are covered by source inspection plus automated checks.

### Gaps Summary

No gaps found. The phase goal is achieved: a selected ROCm Docker Target now carries an auditable PyTorch ROCm dependency policy, the dependency stack is classified before clean validation claims, unsupported and mixed-version states are diagnostic blockers rather than benchmark failures, and the Docker wrapper blocks illegal dependency states before build/run by default while preserving a non-authoritative debug path.

---

_Verified: 2026-05-28T09:48:29Z_
_Verifier: the agent (gsd-verifier)_
