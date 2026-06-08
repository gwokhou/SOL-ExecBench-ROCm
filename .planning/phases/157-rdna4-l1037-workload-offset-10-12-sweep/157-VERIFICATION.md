# Phase 157 Verification

## Bounded Real RDNA4 Offset Runs

All runs used bounded `systemd-run --user` invocations with
`MemoryMax=20G`, `MemorySwapMax=1G`, `--timeout 900`,
`--workload-limit 1`, and a per-offset output directory.

## Results

| Offset | Status | Trace Status Counts | Kernel Rows | Kernel Duration ms |
| ---: | --- | --- | ---: | ---: |
| 10 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 20784.702610 |
| 11 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 159123.633200 |
| 12 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 1997.449461 |

No `rocprofv3`, batch, or `eval_driver.py` process remained after the runs.

## Interpretation

Offsets 10-12 are individually profiler-compatible. Offset 11 is another
medium/slow workload slice, while offset 12 is short.
