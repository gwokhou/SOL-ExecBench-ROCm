# High-Coupling Orchestration Refactor Implementation Plan

**Goal:** Reduce the highest-priority coupling hotspots without changing benchmark behavior.

**Architecture:** Keep public import paths stable. Extract focused helper modules from orchestration files, then leave compatibility wrappers in place where tests or external scripts import old names.

**Scope:**
- Split dataset-script workload sharding helpers out of `scripts/run_dataset.py`.
- Split CLI evaluation no-trace diagnostics helpers out of `src/sol_execbench/cli/evaluation/command.py`.
- Split ROCm architecture detection and compile-command construction out of `src/sol_execbench/driver/problem_packager.py`.
- Split source and safetensors staging helpers out of `src/sol_execbench/driver/problem_packager.py`.
- Remove remaining CLI package import cycles and add a whole-package cycle regression check.
- Convert `src/sol_execbench/core/dataset/__init__.py` to a lazy compatibility facade.
- Route `src/sol_execbench/driver/templates/eval_driver.py` through a focused driver runtime API facade.
- Split derived AMD score artifact and sidecar resolution out of `src/sol_execbench/core/scoring/amd_score_reports.py`.
- Add an executable coupling guardrail script for SCC, facade import, and P0/P1 limit checks.
- Document the coupling governance model and explicit stop conditions.

**Verification:**
- `uv run pytest tests/sol_execbench/core/dataset/test_dataset_sharding.py tests/sol_execbench/cli/evaluation tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/driver/test_eval_driver.py tests/sol_execbench/cli/test_module_boundaries.py tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py -q`
- `uv run python scripts/check_coupling.py`
- `uv run --with ruff ruff check .`

**Tasks:**
- [x] Add or confirm tests around extracted behavior.
- [x] Extract dataset sharding helpers and re-export wrappers from `scripts/run_dataset.py`.
- [x] Extract no-trace diagnostics helpers and keep `command.py` behavior stable.
- [x] Extract ROCm arch detection and compile command helpers from `ProblemPackager`.
- [x] Extract driver file staging helpers from `ProblemPackager`.
- [x] Add no-internal-cycles regression coverage and remove CLI init cycle.
- [x] Replace eager dataset package re-exports with lazy compatibility exports.
- [x] Add a generated-driver runtime API and route the eval driver template through it.
- [x] Extract AMD score derived artifact resolution from report assembly.
- [x] Add a coupling guardrail script and wire it into boundary tests.
- [x] Document coupling governance and stop conditions.
- [x] Run targeted tests and lint.
