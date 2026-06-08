# Phase 136 Verification

## Commands

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/run_dataset.py src/sol_execbench/core/dataset tests/sol_execbench/test_long_tail_exclusions.py tests/sol_execbench/test_run_dataset_execution_closure.py
```

Result: passed.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py::test_long_tail_exclusions_filter_plain_dataset_and_write_closure -q
```

Result: 1 passed.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_dataset_run_closure.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_long_tail_exclusions.py -q
```

Result: 112 passed.

```bash
git diff --check
```

Result: passed.

## Notes

- `tests/examples/test_rocm_cli_paths.py -q -k long_tail` is ROCm-marked and
  skipped on systems without the `requires_rocm` environment. The CPU-safe
  coverage above verifies the runner accounting path without GPU access.
- No long-running RDNA4 validation process was started in this phase.
