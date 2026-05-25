# 58-03 Summary: Runtime Evidence Capability and Guardrails

**Completed:** 2026-05-25
**Status:** Complete

## Delivered

- Added optional evaluator contract capability `runtime.evidence.v1`.
- Preserved evaluator contract schema version
  `sol_execbench.evaluator_contract.v1` and contract version `1.0`.
- Extended contract tests to assert the optional capability and version
  stability.
- Extended public trace guardrails and trace docs to keep v1.13 environment
  snapshot evidence outside canonical trace JSONL.
- Restored a v1.4 compatibility inventory guardrail phrase required by the
  public contract test suite.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py -q
```

Result: `44 passed, 1 skipped in 4.51s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/environment.py src/sol_execbench/core/__init__.py src/sol_execbench/core/data/contract.py tests/sol_execbench/test_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py
```

Result: `All checks passed!`.

