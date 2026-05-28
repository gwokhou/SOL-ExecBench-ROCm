---
plan_id: 82-01
status: completed
completed_at: 2026-05-28
---

# 82-01 Summary

## Completed

- Expanded `docs/CLAIMS.md` with ROCm Compatibility Matrix claim boundaries,
  Target/requested versus observed evidence semantics, Docker container
  user-space validation boundaries, and mixed-version debug override authority
  limits.
- Expanded `docs/TESTING.md` with CPU-safe ROCm Matrix guardrail commands,
  wrapper syntax checks, docs lint command, and marker-gated live ROCm guidance.
- Added `tests/sol_execbench/test_rocm_matrix_docs.py` to guard critical docs
  wording around native-host claims, Target identity, debug overrides, CPU-safe
  guardrails, and live marker-gated validation.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_matrix_docs.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_dependency_matrix_cli.py tests/sol_execbench/test_run_docker_dependency_preflight.py tests/sol_execbench/test_runtime_evidence_reports.py tests/sol_execbench/test_run_docker_runtime_evidence.py tests/sol_execbench/test_rocm_matrix_docs.py -q`
- `bash -n scripts/run_docker.sh`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check docs/CLAIMS.md docs/TESTING.md tests/sol_execbench/test_rocm_matrix_docs.py`
