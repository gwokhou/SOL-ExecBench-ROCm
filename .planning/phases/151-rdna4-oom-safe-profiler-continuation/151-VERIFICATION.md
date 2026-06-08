# Phase 151 Verification

## CPU-Safe Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_rdna4_profiler_timing_batch.py \
  tests/sol_execbench/test_profiler_timing_coverage.py -q
```

Result: `16 passed`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check \
  scripts/run_rdna4_profiler_timing_batch.py \
  tests/sol_execbench/test_rdna4_profiler_timing_batch.py
```

Result: `All checks passed!`

## Real RDNA4 Continuation

Known OOM/codex-kill suspects were classified with blocked sidecars:

```bash
TMPDIR=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/out/tmp-profiler \
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_rdna4_profiler_timing_batch.py \
  --output-dir out/rdna4-profiler-backed-timing-full-20260608 \
  --mark-blocked-problem L2/008_moe_sparse_routing_and_dispatch \
  --mark-blocked-problem L2/016_moe_expert_mlp_with_load_balancing \
  --mark-blocked-only --resume
```

Result: exit `1` as expected because `profiler_blocked` contributes to batch
failure accounting; no profiler/eval process was launched in mark-only mode.

A bounded real target was then run through a transient user systemd service:

```bash
systemd-run --user --wait --pipe --same-dir \
  -p MemoryMax=20G -p MemorySwapMax=1G \
  -E TMPDIR=/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm/out/tmp-profiler \
  -E UV_CACHE_DIR=/tmp/uv-cache \
  /usr/bin/bash -lc 'uv run scripts/run_rdna4_profiler_timing_batch.py \
    --timeout 300 \
    --output-dir out/rdna4-profiler-backed-timing-full-20260608 \
    --temp-dir out/tmp-profiler --resume \
    --only-problem L1/031_repeat_kv_attention_matmul'
```

Result: exit `1`; the target wrote a `profiler_blocked` sidecar with
`trace_status_counts={"PASSED": 7}` and
`failure_reason="rocprofv3 command failed with exit code 1"`.

Final coverage after the OOM-safe continuation:

```json
{
  "problem_denominator": 235,
  "profiler_backed_problems": 60,
  "partial_profiler_backed_problems": 9,
  "profiler_blocked_problems": 3,
  "fallback_timing_problems": 49,
  "readiness_blocked_problems": 114,
  "profiler_backed_coverage_pct": 25.5319,
  "full_profiler_backed_timing_coverage": false
}
```

No `rocprofv3`, batch, or `eval_driver.py` process remained after the bounded
run.

## Follow-Up Bounded Continuation

Three additional L1 fallback targets were run one-by-one under the same
bounded systemd pattern:

- `L1/026_video_patch_embedding_projection`: classified `profiler_blocked`
  after `rocprofv3` exit code 1 with `INVALID_REFERENCE=2` and
  `RUNTIME_ERROR=1`.
- `L1/028_hybrid_attention_mask_preparation`: classified `profiler_blocked`
  after `rocprofv3` exit code 1 with `PASSED=2`.
- `L1/037_flux_feedforward_gelu_approximate`: classified `profiler_blocked`
  after the 300 second profiler timeout.

Coverage after this follow-up:

```json
{
  "problem_denominator": 235,
  "profiler_backed_problems": 60,
  "partial_profiler_backed_problems": 9,
  "profiler_blocked_problems": 6,
  "fallback_timing_problems": 46,
  "readiness_blocked_problems": 114,
  "profiler_backed_coverage_pct": 25.5319,
  "full_profiler_backed_timing_coverage": false
}
```

No profiler, batch, or `eval_driver.py` process remained after the follow-up
runs.
