---
status: complete
---

# Summary

Completed the close-partials/ready-missing quick pass and promoted the resulting
coverage into the canonical RDNA4 merged artifacts.

## Execution

- Closed the remaining ready-missing set with workload-sharded kernel-only
  `rocprofv3` timing.
- Resumed after the OOM incident using the OOM-safe compact row path.
- All 49 ready-missing targets now have problem-level timing sidecars.
- The resumed batch processed the final 11 targets:
  - `profiler_backed`: 4
  - `partial_profiler_backed`: 2
  - `profiler_blocked`: 5

Full 49-target batch status:

- `profiler_backed`: 33
- `partial_profiler_backed`: 9
- `profiler_blocked`: 7

After merging with prior evidence and readiness classification, the canonical
coverage is:

- `profiler_backed`: `130 / 235` (`55.3191%`)
- `partial_profiler_backed`: `1`
- `ready_missing_profiler_timing`: `0`
- `reference_oom_blocked`: `57`
- `readiness_blocked`: `41`
- `profiler_blocked`: `6`
- `fallback_timing`: `0`

## Promoted Artifacts

Updated canonical files under:

`out/rdna4-validation-reeval-20260613-latest/merged/`

- `evaluation-summary.json`
- `evaluation-summary.md`
- `profiler-timing-coverage-summary.json`
- `profiler-timing-coverage.json`
- `profiler-timing-blocker-ledger.json`
- `sharded-closure-audit.json`

Latest staging recompute:

`out/rdna4-validation-reeval-20260613-latest/coverage-with-close-partials-ready-missing/`

## Remaining Focus

Remaining sharded closure targets after merge:

- `L2/041_kv_shared_attention_with_dual_rope`: partial, `15/16`, blocked by
  one `INVALID_REFERENCE`.
- `L1/044_moe_expert_computation`: profiler-blocked, all workloads blocked by
  timing input preflight capacity.
- `L2/012_moe_expert_batched_execution_with_capacity_factor`: profiler-blocked,
  all workloads blocked by timing input preflight capacity.
- `L2/024_moe_expert_parallel_execution`: profiler-blocked, all workloads
  blocked by timing input preflight capacity.
- `L2/025_moe_expert_parallel_execution_backward`: profiler-blocked, all
  workloads blocked by timing input preflight capacity.
- `L2/026_moe_expert_parallel_execution_with_weighted_aggregation`:
  profiler-blocked, all workloads blocked by timing input preflight capacity.
- `L2/047_moe_training_token_repeat_and_expert_computation`: profiler-blocked,
  all workloads blocked by timing input preflight capacity.

## Verification

- `uv run python -m py_compile scripts/run_rdna4_profiler_timing_batch.py src/sol_execbench/core/bench/rocm_profiler.py`
- `uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py`
- `uv run pytest tests/sol_execbench/test_rocm_profiler.py`
- `uv run pytest tests/sol_execbench/core/data/test_workload.py`
- `uv run --with ruff ruff check scripts/run_rdna4_profiler_timing_batch.py src/sol_execbench/core/bench/rocm_profiler.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py tests/sol_execbench/test_rocm_profiler.py`
- Merged summary/coverage/audit consistency check.
