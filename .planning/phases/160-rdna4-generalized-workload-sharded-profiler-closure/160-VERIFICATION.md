---
status: complete
---

# Phase 160 Verification

## CPU-Safe Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rdna4_profiler_sharded_closure.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py tests/sol_execbench/test_profiler_timing_coverage.py -q
```

Result: `25 passed`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/run_rdna4_profiler_sharded_closure.py scripts/run_rdna4_profiler_timing_batch.py tests/sol_execbench/test_rdna4_profiler_sharded_closure.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py tests/sol_execbench/test_profiler_timing_coverage.py
```

Result: `All checks passed!`.

## Real Audit

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_rdna4_profiler_sharded_closure.py \
  --dataset-root data/SOL-ExecBench/benchmark \
  --output-dir out/rdna4-profiler-sharded-closure-audit-20260608 \
  --timing-evidence-dir out/rdna4-profiler-workload-aggregate-20260608-v2/timing \
  --timing-evidence-dir out/rdna4-profiler-backed-timing-full-20260608/timing \
  --timing-evidence-dir out/rdna4-timing-evidence/timing
```

Result: 14 targets selected: 9 partial, 5 blocked.

## Real Sharded Closure Attempt

Ran:

```bash
systemd-run --user --wait --pipe --same-dir -p MemoryMax=20G -p MemorySwapMax=1G \
  uv run scripts/run_rdna4_profiler_timing_batch.py \
  --dataset-root data/SOL-ExecBench/benchmark \
  --output-dir out/rdna4-profiler-sharded-closure-l1026-20260608 \
  --replacement-timing-dir out/rdna4-profiler-sharded-closure-l1026-20260608/timing \
  --source-timing-dir out/rdna4-profiler-workload-aggregate-20260608-v2/timing \
  --source-timing-dir out/rdna4-profiler-backed-timing-full-20260608/timing \
  --source-timing-dir out/rdna4-timing-evidence/timing \
  --only-problem L1/026_video_patch_embedding_projection \
  --workload-sharded \
  --timeout 900 \
  --temp-dir out/rdna4-profiler-sharded-closure-l1026-20260608/tmp
```

Result:

- systemd service result: `success`
- selected targets: 1
- status: `partial_profiler_backed`
- expected workloads: 4
- full workload coverage: false
- final coverage with this evidence: 61 profiler-backed, 10 partial, 4 blocked

No residual `rocprofv3`, `eval_driver.py`, or profiler batch processes were
running after verification.
