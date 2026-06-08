# Phase 156 Verification

## Bounded Real RDNA4 Offset Runs

All runs used bounded `systemd-run --user` invocations with
`MemoryMax=20G`, `MemorySwapMax=1G`, `--timeout 900`,
`--workload-limit 1`, and a per-offset output directory.

## Results

| Offset | Status | Trace Status Counts | Kernel Rows | Kernel Duration ms |
| ---: | --- | --- | ---: | ---: |
| 7 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 80301.295208 |
| 8 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 42962.327751 |
| 9 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 316477.583415 |

No `rocprofv3`, batch, or `eval_driver.py` process remained after the runs.

## Interpretation

Offsets 7-9 are individually profiler-compatible. Offset 9 is another slow
workload slice, similar to offset 6, and requires a timeout above 300 seconds.
