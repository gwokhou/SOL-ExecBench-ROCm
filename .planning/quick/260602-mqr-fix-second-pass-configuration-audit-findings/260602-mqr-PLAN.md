---
quick_id: 260602-mqr
slug: fix-second-pass-configuration-audit-findings
status: completed
created_at: "2026-06-02T08:34:48Z"
---

# Quick Task 260602-mqr: Fix Second-Pass Configuration Audit Findings

## Goal

Fix configuration issues found during the second-pass project configuration
audit.

## Plan

1. Make local pre-commit hooks run with `uv run --locked`.
2. Preserve Ruff excludes for explicitly passed hook files.
3. Keep Docker runtime images from installing development dependency groups.
4. Restrict ROCm wheel markers to Linux x86_64.
5. Sync documentation and guardrail tests with the updated configuration.
6. Regenerate `uv.lock` and run focused validation.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv lock`
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`
- `PRE_COMMIT_HOME=/tmp/pre-commit-cache UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit validate-config`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --locked ruff check .`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --locked ty check`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --locked ruff check --force-exclude examples/pytorch/gemma3_swiglu/kernel.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_project_configuration.py tests/sol_execbench/test_run_docker_matrix_script.py tests/docker/dependencies/test_python_dependencies.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_rocm_matrix_docs.py tests/sol_execbench/test_research_release_docs.py -q`
