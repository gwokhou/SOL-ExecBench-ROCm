# 60-01 Summary: Diagnostics CLI and Preflight Checks

**Completed:** 2026-05-25
**Status:** Complete

## Delivered

- Added `EnvironmentDiagnostics` and `EnvironmentCheckResult` JSON models.
- Added explicit PyTorch ROCm smoke checks for runtime availability, simple
  device memory behavior, and HIP-backed event timing.
- Added `sol-execbench doctor --json`, runnable without a problem directory or
  solution.
- Documented doctor output and the opt-in environment snapshot sidecar.
- Preserved benchmark CLI defaults and trace JSONL behavior.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_environment_snapshot.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_contract.py -q
```

Result: `52 passed, 1 skipped in 4.57s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/environment.py src/sol_execbench/core/__init__.py src/sol_execbench/cli/main.py tests/sol_execbench/test_environment_snapshot.py tests/sol_execbench/test_cli_environment_snapshot.py
```

Result: `All checks passed!`.

