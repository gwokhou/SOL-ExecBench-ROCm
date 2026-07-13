# Phase 131 Summary: Dataset License and Provenance Policy

**Completed:** 2026-06-04
**Status:** Complete

## Delivered

- Extended `provenance.toml` with dataset source, license, provenance, and
  redistribution classes for NVIDIA SOL-ExecBench, FlashInfer Trace, generated
  local migration artifacts, and project-owned ROCm code.
- Added `scripts/check_dataset_redistribution.py` for CPU-safe staged-path and
  release-bundle checks.
- Integrated restricted dataset bundle scanning into
  `scripts/check_prerelease_readiness.py`.
- Updated `docs/user/provenance.md`, `docs/user/compliance.md`, and
  `docs/internal/prerelease_artifact_bundle.md` with NVIDIA Evaluation Dataset License
  and Apache-2.0 FlashInfer Trace boundaries.
- Added tests covering policy shape, checker behavior, and prerelease
  restricted-payload blocking.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_provenance_policy.py tests/sol_execbench/test_dataset_redistribution_policy.py tests/sol_execbench/test_prerelease_readiness.py`
  - `18 passed`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/check_dataset_redistribution.py scripts/check_prerelease_readiness.py tests/sol_execbench/test_provenance_policy.py tests/sol_execbench/test_dataset_redistribution_policy.py tests/sol_execbench/test_prerelease_readiness.py`
  - passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/check_dataset_redistribution.py --path data/SOL-ExecBench/benchmark/L1/problem/definition.json --json`
  - expected blocking exit with `source_id: nvidia_sol_execbench`

## Deferred

- Local migration tooling remains Phase 132.
- Readiness classification and ready subsets remain Phase 133.
- Dataset runner integration and public user workflow docs remain Phase 135.
