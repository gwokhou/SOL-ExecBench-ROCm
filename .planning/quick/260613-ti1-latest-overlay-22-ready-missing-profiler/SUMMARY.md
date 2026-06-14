---
status: complete
---

# Summary

Closed the 22 latest-overlay `ready_missing_profiler_timing` targets with the
optimized representative-workload profiler strategy after the relative `TMPDIR`
rocprofv3 segfault root cause was fixed.

## Results

- Targets attempted: 22
- New timing sidecars written: 22
- Partial profiler-backed: 22
- Profiler-blocked after timeout reruns: 0
- Ready-missing after rebuilt merged coverage: 0

The original full-workload trace strategy was stopped because it produced
multi-GB temporary kernel trace files and timed out on FlashInfer long-tail
workloads. The optimized strategy collects kernel trace evidence for one
representative workload per problem, using smaller workload offsets for timeout
targets.

Timeout rerun closure:

- `Quant/001_fp8_attention_output_projection`: offset 5 passed
- `Quant/002_fp8_attention_qkv_projection`: offset 1 passed
- `Quant/010_fp8_attention_output_projection`: offset 11 passed
- `Quant/016_fp8_multi_latent_attention_qkv_projection`: offset 10 passed
- `Quant/023_fp8_mamba2_ssm_discretization`: offset 3 passed

## Artifacts

- Batch output:
  `out/rdna4-close-ready-missing-22-20260613/profiler-optimized-representative/`
- Replacement timing:
  `out/rdna4-close-ready-missing-22-20260613/profiler-sharded-after-tmpdir-fix/timing/`
- Rebuilt coverage:
  `out/rdna4-close-ready-missing-22-20260613/coverage-after-optimized-reruns-merged/`

## Verification

- `ready_missing_profiler_timing_problems`: 0
- `partial_profiler_backed_problems`: 4
- `profiler_blocked_problems`: 25
- `profiler_backed_problems`: 129
- `profiler_backed_coverage_pct`: 54.8936
- No staging directories remained under the batch temp root.
- No raw rocprofv3 CSV files remained under the batch output.
