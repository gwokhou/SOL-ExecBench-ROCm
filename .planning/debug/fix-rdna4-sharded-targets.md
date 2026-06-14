---
status: resolved
trigger: "Fix the two remaining RDNA4 sharded closure targets"
created: 2026-06-13
updated: 2026-06-13
---

# Debug Session: fix-rdna4-sharded-targets

## Symptoms

- Expected behavior: `L2/035_convnextv2_block_with_grn` and `L2/078_fused_final_layer_upsample_with_adaptive_norm` should aggregate complete all-PASSED workload-sharded profiler evidence and become `profiler_backed`.
- Actual behavior: both remain `partial_profiler_backed` in the sharded closure audit.
- Error messages: `L2/035` has 15 PASSED workload traces and 1 `INCORRECT_NUMERICAL`; `L2/078` has 15 PASSED workload traces and 1 missing/profiler-blocked workload.
- Timeline: observed after strict-isolation short-term RDNA4 closure run on 2026-06-13.
- Reproduction: run the RDNA4 profiler timing coverage and sharded closure scripts against `out/rdna4-short-term-closure-20260613` plus the v135 baseline evidence.

## Current Focus

- hypothesis: One or both failures are caused by profiler harness behavior around workload sharding, not by the coverage script itself.
- test: Inspect per-workload manifests, target problem definitions, generated inputs, and timing sidecars for the failed workload offsets.
- expecting: A concrete fix path that either makes the workload pass under strict-isolation profiling or classifies it as a non-closable blocker without corrupting coverage.
- next_action: done

## Evidence

- `L2/078_fused_final_layer_upsample_with_adaptive_norm` workload 7 passed when rerun with a 2400s timeout, but aggregate import still produced 15/16 because older compacted sidecars had `parsed_rows: []` plus `parsed_rows_compacted: true`. The manifest import path treated those as zero kernel rows despite positive `kernel_duration_ms`.
- The same `L2/078` import set also contained an older failed workload-7 sidecar. The import path was first-wins by workload offset, so a stale failed sidecar could hide newer complete evidence.
- `L2/035_convnextv2_block_with_grn` workload 3 still failed after the workload tolerance alias fix. Reference-vs-reference on the same generated inputs was not self-consistent: match ratio stayed around 64-65% against the 98% required threshold.
- For `L2/035`, conv2d, permute, layernorm, and the first matmul were self-consistent in isolation. A mathematically equivalent NCHW/1x1-conv expression for the two pointwise projections was self-consistent with zero reference-vs-reference error on the failing workload.
- Strict-isolation rerun after applying the scoped RDNA4 reference override produced `PASSED` for `L2/035` workload 3 with max absolute and relative correctness error 0.
- Final coverage using the two new aggregate timing dirs reports 90/235 profiler-backed problems, 38.2979% coverage, 0 partial profiler-backed problems, and 0 sharded closure targets.

## Eliminated

- Workload tolerance spelling mismatch was real (`required_match_ratio` vs `required_matched_ratio`) and fixed, but it did not close `L2/035` by itself.
- `torch.backends.cuda.matmul.allow_tf32=False`, `torch.backends.cudnn.allow_tf32=False`, and `torch.set_float32_matmul_precision("highest")` did not improve `L2/035` reference self-consistency.
- `ROCBLAS_DEFAULT_ATOMICS_MODE=0` was not usable in this environment because PyTorch initialized with `No HIP GPUs are available`.

## Resolution

- root_cause: `L2/078` was blocked by workload-sharded import logic that discarded compacted-but-valid kernel summaries and allowed stale failed sidecars to win by directory order. `L2/035` was blocked by a ROCm reference-oracle instability for the large ConvNeXtV2 GRN workload when staged through the original channels-last matmul reference.
- fix: Treat compacted sidecars with positive `kernel_duration_ms` as importable kernel evidence, prefer complete sidecars over incomplete duplicates for the same workload offset, preserve the dataset tolerance alias, and apply a scoped RDNA4 reference override for `L2/035_convnextv2_block_with_grn` that uses equivalent NCHW 1x1 conv pointwise projections.
- verification: `uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py tests/sol_execbench/core/data/test_workload.py` passes with 41 tests. Strict-isolation workload rerun passed for `L2/035` workload 3. Aggregates for both targets are `profiler_backed` with 16/16 PASSED. Final sharded closure audit target count is 0.
- files_changed: `scripts/run_rdna4_profiler_timing_batch.py`, `src/sol_execbench/core/data/workload.py`, `tests/sol_execbench/test_rdna4_profiler_timing_batch.py`, `tests/sol_execbench/core/data/test_workload.py`
