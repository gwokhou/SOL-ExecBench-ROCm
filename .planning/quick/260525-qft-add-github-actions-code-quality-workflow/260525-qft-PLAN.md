---
quick_id: 260525-qft
status: complete
created: 2026-05-25
description: Add GitHub Actions code-quality workflow matching hip-playground
---

# Quick Task 260525-qft: Add GitHub Actions Code Quality Workflow

## Goal

Add a remote GitHub Actions quality workflow following hip-playground's
`Python Quality` structure while adapting checks to this repository's `uv`,
Ruff, Ty, and CPU-safe pytest setup.

## Tasks

1. Add `.github/workflows/code-quality.yml` with push and pull request triggers,
   Python 3.12/3.13 matrix, `setup-python`, `setup-uv`, locked install, Ruff,
   Ty, and CPU-safe pytest checks.
2. Keep Docker dependency readiness tests out of default GitHub-hosted CPU CI.
3. Update testing docs to describe the new remote CI workflow, its CPU-safe
   test subset, and the separate ROCm/Docker validation path.

## Verification

- `uv run ruff check .`
- `uv run ty check`
- `uv run pytest tests/sol_execbench --ignore=tests/sol_execbench/driver/test_eval_driver.py --ignore=tests/sol_execbench/test_e2e.py`
- `uv run pytest tests/examples/test_examples.py -k consistency`
