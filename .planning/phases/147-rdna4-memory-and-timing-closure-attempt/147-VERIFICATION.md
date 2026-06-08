---
phase: 147
title: RDNA4 memory and timing closure attempt
status: verified
verified: 2026-06-08
---

# Phase 147 Verification

## Checks

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/driver/test_eval_driver.py::test_reference_outputs_aliasing_inputs_are_stabilized -q

UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_rocm_profiler.py::test_source_collection_routes_pytorch_to_explicit_fallback \
  tests/sol_execbench/test_rocm_profiler.py::test_pytorch_policy_does_not_masquerade_as_rocprofv3_kernel_activity \
  tests/sol_execbench/test_rocm_profiler.py::test_live_collection_returns_fallback_for_non_rocprofv3_policy -q

UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/driver/test_eval_driver.py -q

UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check \
  src/sol_execbench/driver/templates/eval_driver.py \
  src/sol_execbench/core/bench/rocm_profiler.py \
  tests/sol_execbench/driver/test_eval_driver.py \
  tests/sol_execbench/test_rocm_profiler.py

systemd-run --user --wait --collect --same-dir \
  --property=MemoryMax=24G --property=MemorySwapMax=0 \
  --setenv=UV_CACHE_DIR=/tmp/uv-cache \
  --unit=sol-phase147-memory-l1-026 \
  uv run scripts/run_dataset.py \
  out/rdna4-phase147-memory-v131/problems/L1/026_video_patch_embedding_projection \
  --phase traces --timeout 300 --workload-shard-size 1 \
  --iterations 10 --warmup-runs 2 \
  -o out/rdna4-phase147-memory-v131/run --rerun
```

## Result

- Eval-driver aliasing regression passed.
- Eval-driver subprocess test file passed: 22 tests.
- ROCm profiler fallback policy regression passed: 3 tests.
- Ruff passed for changed Python files.
- Targeted RDNA4 memory retry completed without crashing Codex or the calling
  shell, but the selected workload still failed with HIP OOM inside the
  reference/user function.
- Timing fallback root-cause evidence exists and shows all 121 v1.31 sidecars
  were PyTorch source policies routed to device-event fallback.
- Closure handoff exists and preserves claim boundaries after the attempted
  fixes.
