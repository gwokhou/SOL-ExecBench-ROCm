---
status: complete
quick_id: 260709-kji
slug: f401-noqa
---

# Quick Task 260709-kji Summary

Cleaned unnecessary `F401` suppressions by adding explicit compatibility exports
or replacing import-only availability checks.

Remaining `F401` suppression:

- `src/sol_execbench/driver/templates/eval_driver.py`: required because selected
  runtime API functions must be present in `globals()` for driver integrity
  checks even when not called directly in the template.

Verification:

- `uv run --with ruff ruff check scripts/run_rdna4_profiler_timing_batch.py src/sol_execbench/core/bench/rocm_profiler/__init__.py src/sol_execbench/core/scoring/solar_derivation/__init__.py src/sol_execbench/core/dataset/readiness/__init__.py src/sol_execbench/core/scoring/solar_derivation/parsing.py src/sol_execbench/core/scoring/amd_sol/__init__.py src/sol_execbench/core/scoring/amd_sol/v2.py src/sol_execbench/driver/eval_runtime_api.py src/sol_execbench/driver/templates/eval_driver.py tests/sol_execbench/driver/test_eval_driver.py`
- `uv run --with ruff ruff check . --select PLC0414`
- `uv run pytest tests/sol_execbench/cli/test_module_boundaries.py tests/sol_execbench/core/dataset/test_dataset_inventory_readiness.py tests/sol_execbench/core/scoring/test_solar_derivation_contract.py tests/sol_execbench/core/scoring/test_solar_derivation_evidence.py tests/sol_execbench/core/scoring/test_solar_derivation_family_modeling.py tests/sol_execbench/core/scoring/test_amd_sol_bounds.py tests/sol_execbench/core/scoring/test_amd_sol_v2.py tests/sol_execbench/core/bench/test_rocm_profiler.py tests/sol_execbench/driver/test_eval_driver.py`
