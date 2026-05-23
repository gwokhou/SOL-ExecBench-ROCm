---
phase: 52-dataset-runner-and-public-contract-closure
fixed_at: 2026-05-23T11:40:46Z
review_path: .planning/phases/52-dataset-runner-and-public-contract-closure/52-REVIEW.md
iteration: 1
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 52: Code Review Fix Report

**Fixed at:** 2026-05-23T11:40:46Z
**Source review:** `.planning/phases/52-dataset-runner-and-public-contract-closure/52-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 2
- Fixed: 2
- Skipped: 0

## Fixed Issues

### CR-01: Sidecar filenames allow path traversal outside the requested output directory

**Files modified:** `scripts/run_dataset.py`, `tests/sol_execbench/test_run_dataset_amd_score.py`
**Commit:** 689a110
**Applied fix:** Added a deterministic safe sidecar stem helper and used it for both generated SOLAR derivation and AMD SOL v2 sidecar filenames. Safe identifiers keep their existing filenames; path-shaped or otherwise unsafe identifiers are normalized with a stable digest suffix so sidecars remain inside the requested directories.

### WR-01: New sidecar tests only cover safe fixture identifiers

**Files modified:** `tests/sol_execbench/test_run_dataset_amd_score.py`
**Commit:** 689a110
**Applied fix:** Added a regression test using traversal-shaped definition and workload identifiers, with pre-created outside target directories, proving both sidecar types are written only under the requested sidecar directories and public evidence references point to those safe paths.

## Verification

- `UV_CACHE_DIR=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/.uv-cache UV_PROJECT_ENVIRONMENT=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/.venv uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py -k "sidecar or traversal or solar or amd_sol" -n 0 -x` - passed, 3 passed, 6 deselected.
- `UV_CACHE_DIR=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/.uv-cache UV_PROJECT_ENVIRONMENT=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/.venv uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_v1_9_validation_closure.py tests/sol_execbench/test_solar_derivation_evidence.py -n 0` - passed, 169 passed.
- `UV_CACHE_DIR=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/.uv-cache UV_PROJECT_ENVIRONMENT=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/.venv uv run --with ruff ruff check scripts/run_dataset.py tests/sol_execbench/test_run_dataset_amd_score.py` - passed.

---

_Fixed: 2026-05-23T11:40:46Z_
_Fixer: the agent (gsd-code-fixer)_
_Iteration: 1_
