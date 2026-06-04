# Phase 135 Summary: Dataset Runner Integration and Public Guardrails

**Completed:** 2026-06-04
**Status:** Complete

## Delivered

- Enriched execution closure provenance with migration manifest source,
  manifest checksum, migration kind, source revision, license boundary,
  manifest denominator summary, readiness summary, ready-subset summary, and
  sidecar source refs.
- Extended closure records with readiness class, blocker codes/types, and
  readiness evidence refs so skipped, blocked, and not-ready workloads remain
  visible in denominator reporting.
- Preserved idempotent runner behavior while treating Phase 132/133 manifest,
  readiness, ready-subset, solution mode, runtime config, git commit, and
  requested evidence changes as stale provenance.
- Made ready-subset runs with no runnable workloads write deterministic
  `summary.json` and closure records for readiness-blocked workloads.
- Updated the cookbook with public-safe local migration, ready-subset runner,
  NVIDIA redistribution, FlashInfer provenance, and CDNA3/CDNA4/low-precision
  claim-boundary guidance.
- Added CPU-safe tests for runner integration, public guardrails, readiness
  filtering, idempotent closure summaries, and migrated manifest metadata.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_dataset_run_closure.py tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_dataset_migration.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_dataset_redistribution_policy.py tests/sol_execbench/test_prerelease_readiness.py tests/sol_execbench/test_public_prerelease_docs.py`
  - `74 passed`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/run_dataset.py src/sol_execbench/core/dataset/run_closure.py src/sol_execbench/core/dataset/execution_closure.py tests/sol_execbench/test_dataset_run_closure.py tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_public_prerelease_docs.py`
  - passed
- `git diff --check`
  - passed

## Deferred

- Real CDNA3/MI300X and CDNA4 full-suite hardware validation remains deferred.
- FlashInfer CUDA-kernel ROCm tuning and performance comparison remains
  deferred.
- Public redistribution of NVIDIA SOL-ExecBench original or migrated dataset
  payloads remains forbidden.
