---
status: complete
---

# Summary

Refactored the first set of high-coupling orchestration hotspots while preserving existing public and private compatibility entry points.

Changed:
- Moved workload prefix and shard-path helpers from `scripts/run_dataset.py` into `sol_execbench.core.dataset.sharding`.
- Moved no-trace diagnostics sidecar helpers from `sol_execbench.cli.evaluation.command` into `sol_execbench.cli.evaluation.diagnostics`.
- Moved ROCm gfx detection and HIP offload flag injection from `sol_execbench.driver.problem_packager` into `sol_execbench.driver.build_config`.
- Moved solution source and safetensors input staging from `sol_execbench.driver.problem_packager` into `sol_execbench.driver.staging`.
- Added dataset sharding tests for workload file prefixing and shard path creation.
- Removed the remaining CLI package import cycle and added whole-package strongly connected component coverage.
- Converted `sol_execbench.core.dataset` to a lazy compatibility facade so internal imports do not eagerly couple all dataset subsystems.
- Added `sol_execbench.driver.eval_runtime_api` and routed the generated `eval_driver.py` template through that stable runtime surface.
- Extracted derived AMD score sidecar/artifact resolution into `sol_execbench.core.scoring.amd_score_derived_artifacts`.
- Added `scripts/check_coupling.py` to make SCC, facade import, and P0/P1 boundary checks directly executable.
- Added `docs/internal/coupling_governance.md` to define allowed high-inbound model modules, facade policy, orchestration policy, and stop conditions.
- Added tests that verify the dataset facade is lazy and that the coupling guardrail script passes.

Final coupling spot-check:
- `sol_execbench.core.dataset`: out=0.
- `sol_execbench.driver.templates.eval_driver`: out=1.
- `sol_execbench.core.scoring.amd_score_reports`: out=6.
- Internal import SCCs: none.

Verification:
- `uv run pytest tests/sol_execbench/core/dataset/test_dataset_sharding.py tests/sol_execbench/cli/evaluation tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/driver/test_eval_driver.py tests/sol_execbench/cli/test_module_boundaries.py tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py -q` passed: 148 passed, 1 skipped.
- `uv run python scripts/check_coupling.py` passed: no cycles, no facade import violations, no limit failures.
- `uv run --with ruff ruff check .` passed.
