# Phase 153 Verification

## Bounded Real RDNA4 Offset Runs

All runs used the same bounded systemd pattern:

```bash
systemd-run --user --wait --pipe --same-dir \
  -p MemoryMax=20G -p MemorySwapMax=1G \
  -E TMPDIR=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/out/tmp-profiler \
  -E UV_CACHE_DIR=/tmp/uv-cache \
  /usr/bin/bash -lc 'uv run scripts/run_rdna4_profiler_timing_batch.py \
    --timeout 300 \
    --output-dir <per-offset-output-dir> \
    --temp-dir out/tmp-profiler --resume \
    --only-problem L1/037_flux_feedforward_gelu_approximate \
    --workload-offset <offset> --workload-limit 1'
```

## Results

| Offset | Status | Trace Status Counts | Kernel Rows | Kernel Duration ms |
| ---: | --- | --- | ---: | ---: |
| 0 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 159006.716407 |
| 1 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 20785.512748 |
| 2 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 159034.994906 |
| 3 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 40698.814771 |

No `rocprofv3`, batch, or `eval_driver.py` process remained after the sweep
slice runs.

## Interpretation

Offsets 0-3 are individually profiler-compatible. The full-problem timeout for
`L1/037` is therefore likely caused by later workload offsets or by aggregate
runtime across all 16 workloads under one profiler invocation.
