---
phase: 78-matrix-contract-and-claim-guardrails
reviewed: 2026-05-28T05:41:27Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - src/sol_execbench/core/compatibility.py
  - tests/sol_execbench/test_rocm_compatibility_matrix.py
  - tests/sol_execbench/test_matrix_claim_guardrails.py
  - tests/sol_execbench/test_public_contract_guardrails.py
findings:
  critical: 1
  warning: 2
  info: 0
  total: 3
status: issues_found
---

# Phase 78: Code Review Report

**Reviewed:** 2026-05-28T05:41:27Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Reviewed the strict ROCm compatibility Matrix Entry contract and guardrail tests. The primary defect is that claim flags are only partially coupled to status, scope, and evidence, allowing entries to serialize clean validation claims that the recorded evidence does not support. The aggregate report and artifact reference models also accept internally inconsistent payloads.

## Critical Issues

### CR-01: Claim flags can overstate validation evidence

**File:** `src/sol_execbench/core/compatibility.py:298`

**Issue:** `MatrixEntry._validate_claim_boundaries` validates `host_validated` evidence, but it never rejects unsupported `container_user_space_validated=true` on a native-host-only entry and never rejects `container_validated` on a `native_host` target. The fallback classifier then copies those claims into an allowed benchmark decision at lines 491-499. A payload can therefore be accepted and classified with `benchmark_allowed=True`, `container_user_space_validated=True`, and/or `native_host_validated=True` even when the status/scope/evidence combination does not prove those claims. This directly weakens the Phase 78 guardrail that compatibility entries must not overclaim Docker/native validation.

**Fix:**
```python
@model_validator(mode="after")
def _validate_claim_boundaries(self) -> MatrixEntry:
    target = self.target
    claims = self.claim_boundary

    if self.status is MatrixCompatibilityStatus.CONTAINER_VALIDATED:
        if target.validation_scope is not MatrixValidationScope.CONTAINER_USER_SPACE:
            raise ValueError("container_validated requires container_user_space validation scope.")
        if self.observed.container is None:
            raise ValueError("container_validated requires observed container evidence.")
        if not claims.container_user_space_validated:
            raise ValueError("container_validated requires container_user_space_validated=true.")
        if claims.native_host_validated:
            raise ValueError("container_validated cannot set native_host_validated=true.")

    if self.status is MatrixCompatibilityStatus.HOST_VALIDATED:
        if claims.container_user_space_validated:
            raise ValueError("host_validated cannot set container_user_space_validated=true.")
        # keep existing native-host evidence checks

    if claims.container_user_space_validated and self.observed.container is None:
        raise ValueError("container_user_space_validated requires observed container evidence.")
    if claims.native_host_validated and self.observed.host is None:
        raise ValueError("native_host_validated requires observed host evidence.")

    return self
```

## Warnings

### WR-01: Report status counts can disagree with entries

**File:** `src/sol_execbench/core/compatibility.py:367`

**Issue:** `RocmCompatibilityMatrixReport.status_counts` is user-supplied and unvalidated. The report can serialize negative counts, missing statuses, or counts that disagree with `entries`, which undermines the aggregate compatibility matrix contract for downstream consumers.

**Fix:** Add a report-level validator that derives expected counts from `entries`, rejects negative values, and either fills `status_counts` automatically or raises when supplied counts differ.

### WR-02: Artifact references can point nowhere

**File:** `src/sol_execbench/core/compatibility.py:98`

**Issue:** `MatrixArtifactReference` accepts entries where both `path` and `uri` are `None`, even though artifacts are supposed to support a Matrix Entry with a local path or remote URI. Such references are not resolvable and can make evidence trails appear present when no artifact location was recorded.

**Fix:** Add a model validator requiring at least one of `path` or `uri`:
```python
@model_validator(mode="after")
def _require_location(self) -> MatrixArtifactReference:
    if self.path is None and self.uri is None:
        raise ValueError("MatrixArtifactReference requires path or uri.")
    return self
```

---

_Reviewed: 2026-05-28T05:41:27Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
