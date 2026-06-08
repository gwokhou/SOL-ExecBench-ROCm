# Phase 158 Verification

## Bounded Real RDNA4 Offset Runs

All runs used bounded `systemd-run --user` invocations with
`MemoryMax=20G`, `MemorySwapMax=1G`, `--timeout 900`,
`--workload-limit 1`, and a per-offset output directory.

## Results

| Offset | Status | Trace Status Counts | Kernel Rows | Kernel Duration ms |
| ---: | --- | --- | ---: | ---: |
| 13 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 6045.352309 |
| 14 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 80227.329338 |
| 15 | `partial_profiler_backed` | `{"PASSED": 1}` | 1681 | 239363.072626 |

No `rocprofv3`, batch, or `eval_driver.py` process remained after the runs.

## Interpretation

Offsets 0-15 are individually profiler-compatible. The full-problem timeout for
`L1/037_flux_feedforward_gelu_approximate` is sufficiently explained by
aggregate profiler runtime. The summed kernel duration across the diagnostic
slices is approximately `1842969.159033 ms` (`30.716 min`), before profiler
startup, validation, JSON output, and scheduling overhead.

Further serial offset testing is unnecessary for this problem. The next useful
action is either a single full-problem replacement attempt with a materially
larger timeout, or a workload-shard aggregation design that can promote a
complete set of per-workload profiler slices into a problem-level replacement
only when every expected workload has passed.
