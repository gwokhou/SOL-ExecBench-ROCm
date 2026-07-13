---
quick_id: 260525-qft
status: complete
completed: 2026-05-25
---

# Quick Task 260525-qft Summary

## Outcome

Added a GitHub Actions remote quality workflow modeled after hip-playground's
`Python Quality` workflow. The new workflow runs on pushes and pull requests,
uses Python 3.12 and 3.13, installs through `uv sync --locked --all-groups`,
and runs Ruff, Ty, and CPU-safe pytest checks.

The pytest step excludes eval-driver, full end-to-end execution, and Docker
dependency readiness tests because those require a visible ROCm device or ROCm
container passthrough on this project. The examples coverage is limited to the
existing consistency tests.

## Files Changed

- `.github/workflows/code-quality.yml`
- `docs/user/TESTING.md`

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv sync --locked --all-groups` - passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` - passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check` - passed
- `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/code-quality.yml')); print('workflow yaml ok')"` - passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench --ignore=tests/sol_execbench/driver/test_eval_driver.py --ignore=tests/sol_execbench/test_e2e.py -q` - 850 passed, 57 skipped
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/examples/test_examples.py -k consistency -q` - 14 passed
