---
status: complete
---

# Phase 159 Verification

## CPU-Safe Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py tests/sol_execbench/test_profiler_timing_coverage.py -q
```

Result: `23 passed`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/run_rdna4_profiler_timing_batch.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py tests/sol_execbench/test_profiler_timing_coverage.py
```

Result: `All checks passed!`.

## Real RDNA4 Verification

Ran workload-sharded import-only aggregation for
`L1/037_flux_feedforward_gelu_approximate` using 16 existing real
`rocprofv3` workload slice timing sidecars.

Result:

- `selected_targets`: 1
- `succeeded`: 1
- `status`: `profiler_backed`
- `profiled_workload_count`: 16
- `expected_workload_count`: 16
- `full_workload_coverage`: true
- `kernel_duration_ms`: 1842969.159033

Generated 235-problem coverage summary:

```json
{
  "expected_problem_denominator": 235,
  "fallback_timing_problems": 46,
  "full_profiler_backed_timing_coverage": false,
  "partial_profiler_backed_problems": 9,
  "problem_denominator": 235,
  "profiler_backed_coverage_pct": 25.9574,
  "profiler_backed_problems": 61,
  "profiler_blocked_problems": 5,
  "readiness_blocked_problems": 114,
  "ready_missing_profiler_timing_problems": 0
}
```

No residual `rocprofv3`, `eval_driver.py`, or profiler batch processes were
running after verification.
