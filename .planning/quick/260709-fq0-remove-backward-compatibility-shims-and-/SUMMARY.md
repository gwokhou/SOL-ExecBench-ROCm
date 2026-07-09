---
status: complete
date: 2026-07-09
---

# Summary

Removed backward-compatible CLI/core facade modules for relocated platform,
evidence, reports, and CLI modules. Updated internal imports, tests, scripts,
and the console entry point to use canonical package paths directly.

Removed legacy CUDA/CUPTI timing aliases from `core.bench.timing`; `time_runnable`
now accepts only the ROCm device-event methodology. Tightened timing residue
audits so the old alias names are forbidden rather than allowlisted.

Verification:
- `uv run --with ruff ruff check ...` passed for touched focused paths.
- `uv run pytest tests/sol_execbench/core/bench/test_timing.py tests/sol_execbench/core/dataset/test_rocm_eval_timing_audit.py tests/sol_execbench/core/evidence/test_public_import_facades.py tests/sol_execbench/cli/test_module_boundaries.py` passed: 44 passed, 28 skipped.
