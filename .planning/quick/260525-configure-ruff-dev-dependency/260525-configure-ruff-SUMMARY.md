---
status: complete
quick_id: 260525-configure-ruff
date: 2026-05-25
---

# Quick Task Summary: Configure Ruff Dev Dependency

## Delivered

- Added `ruff>=0.4` to the `dev` dependency group in `pyproject.toml`.
- Updated `uv.lock` so Ruff is installed by `uv sync --all-groups`.
- Updated docs to use `uv run ruff check .` and `uv run ruff format .`.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
```

Result: `All checks passed!`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -n 0 tests/sol_execbench/test_rocm_library_readiness_docs.py tests/sol_execbench/test_public_contract_guardrails.py::test_cli_help_preserves_existing_public_options -q
```

Result: `5 passed in 0.71s`.
