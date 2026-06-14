---
status: completed
quick_id: 260613-fix-l2041-partial
slug: fix-l2041-partial
description: Fix the remaining partial profiler-backed RDNA4 target L2/041
created_at: 2026-06-13T00:00:00+08:00
---

# Quick Task 260613-fix-l2041-partial

## Goal

Close the remaining `partial_profiler_backed` gap:

- `L2/041_kv_shared_attention_with_dual_rope`

## Current Evidence

Canonical merged coverage after the close-partials/ready-missing quick:

- `profiler_backed`: `130 / 235`
- `partial_profiler_backed`: `1`
- `ready_missing_profiler_timing`: `0`

The partial target has `15/16` workloads passing. The failing workload is:

- offset `2`
- uuid `f60ca37e-a4f9-5448-a286-52204370bf1f`
- axes: `batch_size=1`, `seq_len=4096`
- observed status: `INVALID_REFERENCE`

## Plan

1. [x] Reproduce workload offset `2` directly.
2. [x] Inspect `INVALID_REFERENCE` logs to identify whether the reference returns
   invalid numeric values, violates schema, or fails during timing validation.
3. [x] Apply the narrowest fix consistent with benchmark semantics.
4. [x] Rerun `L2/041` workload-sharded timing and merge if it becomes full
   `profiler_backed`.

## Result

Closed.

Root cause:

- The failing workload was not a reference semantics problem.
- The `INVALID_REFERENCE` was caused by an artificially low
  `--subprocess-memory-limit-gib 18` address-space limit. For
  `batch_size=1, seq_len=4096`, PyTorch ROCm reaches the final attention
  value matmul after allocating large intermediates, then fails late in
  `hipblasCreate(handle)` with `HIPBLAS_STATUS_ALLOC_FAILED`.
- Running the original reference without that RLIMIT_AS cap passes.

Evidence:

- Failing reproduction:
  `out/rdna4-fix-l2041-partial-20260613/workload-0002/timing/L2/041_kv_shared_attention_with_dual_rope.timing.json`
- Passing original-reference slice:
  `out/rdna4-fix-l2041-partial-20260613/workload-0002-original-nomemlimit/timing/L2/041_kv_shared_attention_with_dual_rope.timing.json`
- Workload-sharded aggregate:
  `out/rdna4-fix-l2041-partial-20260613/aggregate/timing/L2/041_kv_shared_attention_with_dual_rope.timing.json`
- Updated coverage:
  `out/rdna4-validation-reeval-20260613-latest-plus-l2041/profiler-timing-coverage/coverage-summary.json`

Coverage delta:

- `profiler_backed`: `130 / 235` -> `131 / 235`
- `partial_profiler_backed`: `1` -> `0`
- `ready_missing_profiler_timing`: remains `0`

Validation:

- `uv run python -m py_compile scripts/run_rdna4_profiler_timing_batch.py`
- `uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py -q`
  (`42 passed`)
- `uv run --with ruff ruff check scripts/run_rdna4_profiler_timing_batch.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py`
