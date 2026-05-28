---
phase: 82-validation-workflow-docs-and-ci-guardrails
verified: 2026-05-28T10:40:00Z
status: passed
score: 8/8 must-haves verified
overrides_applied: 0
---

# Phase 82: Validation Workflow, Docs, And CI Guardrails Verification Report

**Phase Goal:** Users can follow documented validation workflows and trust
automated guardrails that prevent ROCm matrix evidence from being overstated.

## Goal Achievement

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Docs explain Docker Matrix Entries as container ROCm user-space validation on recorded host driver/devices. | VERIFIED | `docs/CLAIMS.md` contains explicit Matrix wording and docs tests assert it. |
| 2 | Docs state Docker Matrix Entries do not prove native host ROCm validation. | VERIFIED | `docs/CLAIMS.md` has native-host boundary wording and forbidden-claim language. |
| 3 | Docs explain Target/requested versus observed evidence. | VERIFIED | `docs/CLAIMS.md` defines both concepts and why Target identity is required. |
| 4 | Docs explain mixed-version default blocking and debug override limits. | VERIFIED | `docs/CLAIMS.md` documents probe/smoke-only debug override and denies validation/authority claims. |
| 5 | Docs list CPU-safe guardrail commands. | VERIFIED | `docs/TESTING.md` lists the focused v1.18 Matrix suite, `bash -n`, and Ruff docs checks. |
| 6 | Docs describe marker-gated live ROCm validation. | VERIFIED | `docs/TESTING.md` references `requires_rocm`, `requires_rdna4`, and `requires_cdna3`. |
| 7 | Docs state current host ROCm 7.1.x can be recorded as observed evidence without host reinstall for 7.0.x/7.2.x default validation. | VERIFIED | `docs/TESTING.md` includes this guidance and tests assert it. |
| 8 | Full v1.18 CPU-safe guardrail suite passes. | VERIFIED | `106 passed in 2.90s`; Ruff and shell syntax also pass. |

**Score:** 8/8 truths verified

## Verification Commands

| Command | Result |
|---|---|
| `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_matrix_docs.py -q` | `5 passed` |
| `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_dependency_matrix_cli.py tests/sol_execbench/test_run_docker_dependency_preflight.py tests/sol_execbench/test_runtime_evidence_reports.py tests/sol_execbench/test_run_docker_runtime_evidence.py tests/sol_execbench/test_rocm_matrix_docs.py -q` | `106 passed in 2.90s` |
| `bash -n scripts/run_docker.sh` | PASS |
| `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check docs/CLAIMS.md docs/TESTING.md tests/sol_execbench/test_rocm_matrix_docs.py` | PASS |

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| DOCS-01 | SATISFIED | `docs/CLAIMS.md` Docker Matrix/native-host boundary wording. |
| DOCS-02 | SATISFIED | `docs/CLAIMS.md` Target/requested and observed evidence section. |
| DOCS-03 | SATISFIED | `docs/CLAIMS.md` mixed-version block/debug override authority wording. |
| DOCS-04 | SATISFIED | CPU-safe guardrail suite and docs wording tests. |
| DOCS-05 | SATISFIED | Existing Docker/script tests plus `docs/TESTING.md` guardrail command list. |
| DOCS-06 | SATISFIED | Marker-gated live ROCm guidance in `docs/TESTING.md`. |

## Human Verification Required

None. Live ROCm validation remains marker-gated and optional for this phase.
