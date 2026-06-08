# Phase 152 Verification

## CPU-Safe Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_rdna4_profiler_timing_batch.py \
  tests/sol_execbench/test_profiler_timing_coverage.py -q
```

Result: `18 passed`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check \
  scripts/run_rdna4_profiler_timing_batch.py \
  tests/sol_execbench/test_rdna4_profiler_timing_batch.py
```

Result: `All checks passed!`

## Real RDNA4 Workload Slice

```bash
systemd-run --user --wait --pipe --same-dir \
  -p MemoryMax=20G -p MemorySwapMax=1G \
  -E TMPDIR=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/out/tmp-profiler \
  -E UV_CACHE_DIR=/tmp/uv-cache \
  /usr/bin/bash -lc 'uv run scripts/run_rdna4_profiler_timing_batch.py \
    --timeout 300 \
    --output-dir out/rdna4-profiler-workload-shards-20260608 \
    --temp-dir out/tmp-profiler --resume \
    --only-problem L1/037_flux_feedforward_gelu_approximate \
    --workload-offset 0 --workload-limit 1'
```

Result: success. The sidecar recorded:

```json
{
  "replacement_status": "partial_profiler_backed",
  "profiled_workload_count": 1,
  "expected_workload_count": 16,
  "trace_status_counts": {"PASSED": 1},
  "workload_limit_applied": 1,
  "workload_offset": 0,
  "workload_slice_applied": true,
  "full_workload_coverage": false
}
```

The profiler evidence collected `1681` parsed kernel rows with
`kernel_duration_ms=159006.716407`.

## Claim Boundary Check

Coverage over the diagnostic shard evidence plus fallback timing:

```json
{
  "problem_denominator": 235,
  "profiler_backed_problems": 0,
  "partial_profiler_backed_problems": 1,
  "fallback_timing_problems": 120,
  "readiness_blocked_problems": 114,
  "profiler_backed_coverage_pct": 0.0,
  "full_profiler_backed_timing_coverage": false
}
```

The workload slice is diagnostic only; it does not count as full
profiler-backed timing coverage.
