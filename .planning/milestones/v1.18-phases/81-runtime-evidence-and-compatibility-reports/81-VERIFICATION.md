---
phase: 81-runtime-evidence-and-compatibility-reports
verified: 2026-05-28T10:20:00Z
status: passed
score: 10/10 must-haves verified
overrides_applied: 0
---

# Phase 81: Runtime Evidence And Compatibility Reports Verification Report

**Phase Goal:** Users can collect scoped runtime evidence and review per-Target
and aggregate compatibility reports without changing benchmark semantics.

## Goal Achievement

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Host evidence is separate from container/toolchain evidence. | VERIFIED | `runtime_evidence.py` builds separate `MatrixHostEvidence`, `MatrixContainerEvidence`, and `MatrixToolchainEvidence`; tests assert separate `observed` keys. |
| 2 | Python dependency evidence includes torch, HIP, CUDA compatibility namespace, device availability, and Triton ROCm status. | VERIFIED | `build_dependency_observation()` and Phase 80 `MatrixPythonDependencyEvidence` carry those fields; tests assert the serialized JSON. |
| 3 | GPU evidence includes count, name, gfx architecture, and visible-device environment. | VERIFIED | `collect_gpu_evidence()` supports injected CPU-safe values and env capture; tests assert `gfx1200` plus `HIP_VISIBLE_DEVICES`. |
| 4 | Per-target compatibility JSON sidecars are emitted. | VERIFIED | `write_matrix_entry()` and `python -m sol_execbench.core.runtime_evidence collect-target` write strict Matrix Entry JSON. |
| 5 | Aggregate compatibility matrix JSON includes exact status counts. | VERIFIED | `build_aggregate_report()` validates `status_counts`; tests assert counts for `not_tested`, `runtime_unavailable`, and `mixed_version`. |
| 6 | Setup/runtime, dependency, benchmark correctness, and benchmark performance categories are distinct. | VERIFIED | `RuntimeFailureEvidence.category` is bounded to those four categories and serialized as diagnostic artifacts. |
| 7 | Canonical Trace payloads are not mutated. | VERIFIED | Tests build a `Trace`, generate compatibility evidence, and assert the trace payload and `Trace.model_fields` are unchanged. |
| 8 | Docker wrapper sidecar writing is explicit opt-in. | VERIFIED | `--compatibility-entry`, `--compatibility-matrix`, and env equivalents were added; tests assert default dry-run writes no sidecar. |
| 9 | Blocked dependency/runtime states are represented as sidecar statuses, not benchmark failures. | VERIFIED | Wrapper tests assert mixed dependency and runtime-unavailable cases write compatibility JSON and stop before Docker command text. |
| 10 | Phase 79/80 preflight behavior remains green. | VERIFIED | v1.18 regression suite including Docker matrix and dependency preflight tests passed. |

**Score:** 10/10 truths verified

## Verification Commands

| Command | Result |
|---|---|
| `bash -n scripts/run_docker.sh` | PASS |
| `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_runtime_evidence_reports.py tests/sol_execbench/test_run_docker_runtime_evidence.py tests/sol_execbench/test_run_docker_dependency_preflight.py tests/sol_execbench/test_dependency_matrix_cli.py tests/sol_execbench/test_docker_matrix_preflight.py -q` | `32 passed in 2.26s` |
| `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_dependency_matrix_cli.py tests/sol_execbench/test_run_docker_dependency_preflight.py tests/sol_execbench/test_runtime_evidence_reports.py tests/sol_execbench/test_run_docker_runtime_evidence.py -q` | `101 passed in 2.79s` |
| `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/runtime_evidence.py tests/sol_execbench/test_runtime_evidence_reports.py tests/sol_execbench/test_run_docker_runtime_evidence.py` | PASS |

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| EVID-01 | SATISFIED | Separate host/container/toolchain evidence scopes in Matrix observed evidence. |
| EVID-02 | SATISFIED | Python dependency evidence fields and tests. |
| EVID-03 | SATISFIED | GPU evidence helper and wrapper env propagation tests. |
| EVID-04 | SATISFIED | Per-target sidecar and aggregate report helpers/CLI. |
| EVID-05 | SATISFIED | `RuntimeFailureEvidence` categories and artifact serialization. |
| EVID-06 | SATISFIED | Trace non-mutation tests and explicit opt-in wrapper sidecars. |

## Human Verification Required

None. Phase 81 is covered by CPU-safe unit, CLI, and shell dry-run tests.
