---
quick_id: 260602-mqi
slug: fix-stale-project-configuration-audit-fi
status: completed
created_at: "2026-06-02T08:22:12.943Z"
---

# Quick Task 260602-mqi: Fix Stale Project Configuration Audit Findings

## Goal

Fix stale or inconsistent project configuration found in the configuration
audit.

## Plan

1. Make CI's Python matrix actually drive uv's selected interpreter.
2. Update Docker uv version and the test that guards it.
3. Replace remote Ruff pre-commit hooks with local uv-backed hooks.
4. Restrict ROCm PyTorch wheel markers to Linux.
5. Update hook setup docs wording.
6. Regenerate `uv.lock` and run config, lint/type, and focused tests.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv lock`
- `PRE_COMMIT_HOME=/tmp/pre-commit-cache UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit validate-config`
- `PRE_COMMIT_HOME=/tmp/pre-commit-cache UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit run --hook-stage pre-commit --files tests/docker/dependencies/test_python_dependencies.py tests/sol_execbench/test_run_docker_matrix_script.py`
- `PRE_COMMIT_HOME=/tmp/pre-commit-cache UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit run --hook-stage pre-push --all-files`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_docker_matrix_script.py tests/docker/dependencies/test_python_dependencies.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_docker_matrix_targets.py -q`
