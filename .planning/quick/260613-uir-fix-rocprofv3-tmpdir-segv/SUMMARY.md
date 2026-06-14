---
status: complete
---

# Summary

Root-caused `rocprofv3` exit `-11` to a relative `TMPDIR` inherited by staged profiler subprocesses. With `TMPDIR=out/.../tmp`, the subprocess runs from a staging directory, so ROCm/PyTorch sees a different relative temp path and can segfault during BF16 `torch.randn` under `rocprofv3`.

Fixed the RDNA4 profiler batch runner to pass an absolute staged temp root as `TMPDIR`, and changed profiler batch eval execution from bare `python eval_driver.py` to the active `sys.executable` for deterministic ROCm venv execution.

After the fix:

- The previous single-workload exit `-11` repro for `FlashInfer-Bench/012_gqa_paged_decode_h32_kv4_d128_ps1` becomes profiler-backed and produces a kernel CSV.
- The previous no-CSV repro for `FlashInfer-Bench/020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` also produces a kernel CSV; its remaining status is `partial_profiler_backed` with `INVALID_REFERENCE`/OOM traces, not profiler output absence.

# Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py -q`
- Real ROCm repro with relative `TMPDIR` for `FlashInfer-Bench/012_gqa_paged_decode_h32_kv4_d128_ps1 --workload-limit 1`
- Real ROCm repro with relative `TMPDIR` for `FlashInfer-Bench/020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
