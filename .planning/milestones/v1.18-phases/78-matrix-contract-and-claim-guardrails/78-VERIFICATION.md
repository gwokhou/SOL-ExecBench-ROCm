---
phase: 78-matrix-contract-and-claim-guardrails
verified: 2026-05-28T05:50:23Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 78: Matrix Contract And Claim Guardrails Verification Report

**Phase Goal:** Users and downstream tools can interpret ROCm compatibility Matrix Entries with stable diagnostic semantics and explicit claim boundaries.
**Verified:** 2026-05-28T05:50:23Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can read a `sol_execbench.rocm_compatibility_matrix.v1` report and identify each Matrix Entry's Target, observed evidence, status, reason codes, artifacts, and claim boundaries. | VERIFIED | `src/sol_execbench/core/compatibility.py` defines `ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION`, `MatrixEntry`, `RocmCompatibilityMatrixReport`, `MatrixTarget`, `MatrixObservedEvidence`, `MatrixArtifactReference`, and `MatrixClaimBoundary`; `test_matrix_entry_serializes_target_and_observed_evidence_separately`, `test_matrix_report_contains_entries_and_status_counts`, and `test_matrix_entry_carries_artifacts_and_diagnostic_claim_boundaries` verify serialized payloads. |
| 2 | User can distinguish requested Target values from observed host, container, Python dependency, toolchain, and GPU evidence in every Matrix Entry. | VERIFIED | `MatrixEntry` has separate required `target` and `observed` fields; `MatrixObservedEvidence` separates `host`, `container`, `python_dependency`, `toolchain`, and `gpu`; tests assert requested Target values and observed evidence serialize under distinct objects. |
| 3 | User can rely on the bounded status vocabulary: `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, and `not_tested`. | VERIFIED | `MatrixCompatibilityStatus` contains exactly those six enum values; `test_matrix_status_vocabulary_is_locked` verifies the exact set; execution-decision tests cover mixed-version, wheel-unavailable, runtime-unavailable, and not-tested semantics. |
| 4 | User can verify that compatibility evidence is diagnostic-only and never grants score, paper-parity, or leaderboard authority. | VERIFIED | `MatrixClaimBoundary` and `MatrixExecutionDecision` use `Literal[True]` for diagnostic evidence and `Literal[False]` for score, paper-parity, and leaderboard authority; tests reject attempts to set authority flags true and verify debug override decisions remain non-authoritative. |
| 5 | User never sees Docker container validation described as native host validation; `host_validated` requires direct native-host evidence. | VERIFIED | `MatrixEntry._validate_claim_boundaries` rejects container-scoped `host_validated`, `native_host_validated=true` on container entries, and host validation without direct host ROCm/driver evidence; `test_docker_scope_cannot_serialize_host_validated_status`, `test_docker_container_validated_claims_container_user_space_not_native_host`, and native-host tests cover the guardrails. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/sol_execbench/core/compatibility.py` | Strict compatibility matrix contract plus pure claim/execution guardrails. | VERIFIED | `gsd-sdk query verify.artifacts` passed for both plans. Manual inspection found strict frozen Pydantic models, bounded enums, validation, `to_dict`, status count validation, `build_matrix_entry`, and `classify_matrix_entry_for_execution`. |
| `tests/sol_execbench/test_rocm_compatibility_matrix.py` | CPU-safe schema, vocabulary, serialization, artifact, and authority-flag tests. | VERIFIED | Artifact check passed; tests cover schema version, Target/observed separation, unknown field rejection, status vocabulary, status counts, artifacts, and claim flags. |
| `tests/sol_execbench/test_matrix_claim_guardrails.py` | CPU-safe mixed-version, debug override, host/container claim, and wording guardrails. | VERIFIED | Artifact check passed; tests cover benchmark blocking, debug override limits, unavailable classifications, Docker/native-host separation, and Matrix Entry/Target wording. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Canonical trace/public payload sidecar isolation guardrails. | VERIFIED | Artifact check passed; `test_v1_18_compatibility_matrix_fields_remain_sidecar_only` asserts compatibility matrix keys are absent from canonical `Definition`, `Workload`, and `Trace` payloads. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/sol_execbench/test_rocm_compatibility_matrix.py` | `src/sol_execbench/core/compatibility.py` | Imports Matrix Entry and report contract models. | WIRED | `gsd-sdk query verify.key-links` for plan 78-01 passed. |
| `src/sol_execbench/core/compatibility.py` | `tests/sol_execbench/test_matrix_claim_guardrails.py` | Classification helper assertions. | WIRED | `gsd-sdk query verify.key-links` for plan 78-02 passed. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | `src/sol_execbench/core/data/trace.py` | Canonical trace payload assertions. | WIRED | `gsd-sdk query verify.key-links` for plan 78-02 passed. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `src/sol_execbench/core/compatibility.py` | Pydantic model fields and enum values | Explicit constructor/model validation inputs | Yes | VERIFIED - contract models serialize caller-provided Target, observed evidence, artifacts, claim boundaries, and report counts; no dynamic fetch/store path applies. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Canonical payload key sets | Representative `Definition`, `Workload`, and `Trace` fixtures | Yes | VERIFIED - tests serialize real canonical model payloads and assert compatibility sidecar key spaces are absent. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase 78 contract and guardrail tests pass. | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py -q` | `60 passed in 2.96s` | PASS |
| Declared artifacts exist and are substantive. | `gsd-sdk query verify.artifacts` for `78-01-PLAN.md` and `78-02-PLAN.md` | Plan 78-01: 2/2 passed; Plan 78-02: 3/3 passed. | PASS |
| Declared key links are wired. | `gsd-sdk query verify.key-links` for `78-01-PLAN.md` and `78-02-PLAN.md` | Plan 78-01: 1/1 verified; Plan 78-02: 2/2 verified. | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| None declared for Phase 78. | `find scripts -path '*/tests/probe-*.sh' -type f` | No conventional probe scripts found. | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| MATRIX-01 | 78-01, 78-02 | Diagnostic report contract with explicit Target, observed evidence, status, reason-code, artifact, and claim-boundary fields. | SATISFIED | `MatrixEntry` and `RocmCompatibilityMatrixReport` expose all fields; serialization tests verify payload shape. |
| MATRIX-02 | 78-01 | Stable Target identity with requested ROCm user-space, Docker image/tag, PyTorch ROCm Target, validation scope, and intended GPU architecture. | SATISFIED | `MatrixTarget` defines all required fields; representative tests assert each serialized Target value. |
| MATRIX-03 | 78-01 | Requested Target values are distinguished from observed host, container, Python dependency, toolchain, and GPU evidence. | SATISFIED | Separate `MatrixTarget` and `MatrixObservedEvidence` objects; observed submodels for all required evidence scopes; serialization tests cover the split. |
| MATRIX-04 | 78-01, 78-02 | Bounded status vocabulary. | SATISFIED | `MatrixCompatibilityStatus` has the exact six allowed values; tests lock the set and decision behavior. |
| MATRIX-05 | 78-01, 78-02 | Claim flags keep compatibility evidence diagnostic-only and never grant score, paper-parity, or leaderboard authority. | SATISFIED | Claim and decision authority fields are literal false; tests reject true authority flags and verify non-authoritative debug override decisions. |
| MATRIX-06 | 78-02 | `host_validated` only for direct native-host evidence, not Docker user-space Matrix Entries. | SATISFIED | `MatrixEntry` validation rejects Docker/native overclaims; tests cover container-scoped `host_validated`, native-host evidence requirements, and container wording. |

No orphaned Phase 78 requirements were found: `.planning/REQUIREMENTS.md` maps exactly MATRIX-01 through MATRIX-06 to Phase 78.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `docs/internal/v1_4_compatibility_inventory.md` | 89 | `REWARD_HACK` matched the broad `HACK` scan token. | Info | This is a documented canonical evaluation status, not a debt marker or placeholder. No blocker. |

### Human Verification Required

None. Phase 78 is a CPU-safe schema and pure guardrail phase; all must-haves are verifiable through code inspection and automated tests.

### Gaps Summary

No gaps found. The compatibility Matrix Entry contract, bounded diagnostic semantics, claim-boundary enforcement, mixed-version guardrails, and sidecar-only public contract isolation are implemented and covered by targeted tests.

---

_Verified: 2026-05-28T05:50:23Z_
_Verifier: the agent (gsd-verifier)_
