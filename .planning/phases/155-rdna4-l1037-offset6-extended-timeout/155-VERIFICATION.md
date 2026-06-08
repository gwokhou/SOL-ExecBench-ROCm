# Phase 155 Verification

## Bounded Real RDNA4 Run

```bash
systemd-run --user --wait --pipe --same-dir \
  -p MemoryMax=20G -p MemorySwapMax=1G \
  -E TMPDIR=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/out/tmp-profiler \
  -E UV_CACHE_DIR=/tmp/uv-cache \
  /usr/bin/bash -lc 'uv run scripts/run_rdna4_profiler_timing_batch.py \
    --timeout 900 \
    --output-dir out/rdna4-profiler-workload-shards-20260608-l1037-offset06-timeout900 \
    --temp-dir out/tmp-profiler --resume \
    --only-problem L1/037_flux_feedforward_gelu_approximate \
    --workload-offset 6 --workload-limit 1'
```

Result: success after 5min 19.943s service runtime.

## Sidecar Result

```json
{
  "replacement_status": "partial_profiler_backed",
  "profiled_workload_count": 1,
  "expected_workload_count": 16,
  "trace_status_counts": {"PASSED": 1},
  "workload_offset": 6,
  "workload_limit_applied": 1,
  "workload_slice_applied": true,
  "full_workload_coverage": false
}
```

Profiler evidence:

```json
{
  "backend": "rocprofv3",
  "kernel_duration_ms": 316696.339607,
  "parsed_rows": 1681
}
```

## Interpretation

Offset 6 is a slow workload slice, not a profiler hang. It fails the 300 second
run only because it requires roughly 5.3 minutes under the profiler. No
`rocprofv3`, batch, or `eval_driver.py` process remained after the run.
