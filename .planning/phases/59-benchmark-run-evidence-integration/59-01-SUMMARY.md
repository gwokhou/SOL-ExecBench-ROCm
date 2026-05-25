# 59-01 Summary: Benchmark Run Evidence Sidecar

**Completed:** 2026-05-25
**Status:** Complete

## Delivered

- Added opt-in environment snapshot sidecar writing to the benchmark CLI.
- Supported explicit `SOLEXECBENCH_ENV_SNAPSHOT_PATH`.
- Supported `SOLEXECBENCH_ENV_SNAPSHOT=1` with `--output`, writing
  `<trace-output>.environment.json`.
- Kept default CLI output and trace JSONL unchanged.
- Made snapshot collection failures non-fatal for benchmark correctness and
  exit status.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py -q
```

Result: `49 passed, 1 skipped in 4.52s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/cli/main.py tests/sol_execbench/test_cli_environment_snapshot.py src/sol_execbench/core/environment.py tests/sol_execbench/test_environment_snapshot.py
```

Result: `All checks passed!`.

