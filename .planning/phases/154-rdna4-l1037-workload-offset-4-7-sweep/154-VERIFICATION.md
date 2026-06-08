# Phase 154 Verification

## Bounded Real RDNA4 Offset Runs

All runs used bounded `systemd-run --user` invocations with
`MemoryMax=20G`, `MemorySwapMax=1G`, `--workload-limit 1`, and a per-offset
output directory.

## Results

| Offset | Status | Trace Status Counts | Kernel Rows | Kernel Duration ms | Notes |
| ---: | --- | --- | ---: | ---: | --- |
| 4 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 40653.443967 | Passed diagnostic slice. |
| 5 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 158810.590709 | Passed diagnostic slice. |
| 6 | `profiler_blocked` | `{}` | 0 | n/a | Timed out after 300 seconds before any `PASSED` trace. |

Offset 7 was intentionally not run after offset 6 identified a concrete
single-workload timeout blocker.

No `rocprofv3`, batch, or `eval_driver.py` process remained after the runs.

## Interpretation

Offsets 0-5 are individually profiler-compatible. Offset 6 is the first
identified workload-level timeout blocker for
`L1/037_flux_feedforward_gelu_approximate`.
