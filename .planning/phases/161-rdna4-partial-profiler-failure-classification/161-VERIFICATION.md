---
status: complete
---

# Phase 161 Verification

## CPU-Safe Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rdna4_profiler_partial_failures.py tests/sol_execbench/test_rdna4_profiler_sharded_closure.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py tests/sol_execbench/test_profiler_timing_coverage.py -q
```

Result: `30 passed`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/run_rdna4_profiler_partial_failures.py scripts/run_rdna4_profiler_sharded_closure.py scripts/run_rdna4_profiler_timing_batch.py tests/sol_execbench/test_rdna4_profiler_partial_failures.py tests/sol_execbench/test_rdna4_profiler_sharded_closure.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py tests/sol_execbench/test_profiler_timing_coverage.py
```

Result: `All checks passed!`.

## Real Classification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_rdna4_profiler_partial_failures.py \
  --dataset-root data/SOL-ExecBench/benchmark \
  --output-dir out/rdna4-profiler-partial-failure-classification-20260608 \
  --timing-evidence-dir out/rdna4-profiler-sharded-closure-l1026-20260608/timing \
  --timing-evidence-dir out/rdna4-profiler-workload-aggregate-20260608-v2/timing \
  --timing-evidence-dir out/rdna4-profiler-backed-timing-full-20260608/timing \
  --timing-evidence-dir out/rdna4-timing-evidence/timing
```

Result:

- Partial targets: 10
- `blocked_on_correctness`: 6
- `blocked_on_runtime`: 2
- `blocked_on_mixed_failures`: 2
- No stale `complete_missing_workload_slices.txt` remains after regeneration.
