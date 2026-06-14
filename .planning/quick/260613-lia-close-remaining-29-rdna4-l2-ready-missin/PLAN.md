---
status: complete
quick_id: 260613-lia
slug: close-remaining-29-rdna4-l2-ready-missin
description: Close remaining 29 RDNA4 L2 ready-missing profiler timing targets
created_at: 2026-06-13T07:29:09.201Z
---

# Quick Task 260613-lia: Close Remaining RDNA4 L2 Ready-Missing Profiler Timing

## Goal

Close the remaining 29 `L2/*` problems that are still `ready_missing_profiler_timing` in the latest stage-1 merged coverage artifact.

## Baseline

- Latest coverage root: `out/rdna4-ready-missing-profiler-closure-20260613/coverage-stage1/`
- `profiler_backed_problems`: 121 / 235
- `profiler_backed_coverage_pct`: 51.4894
- `ready_missing_profiler_timing_problems`: 29
- All 29 remaining ready-missing targets are L2.

## Target Set

Target list: `remaining-l2-ready-missing-targets.txt`

Target count: 29

## Execution Strategy

Run targets serially, one L2 problem per batch, using workspace-backed temp and uv cache:

- `TMPDIR=out/rdna4-ready-missing-profiler-closure-20260613/tmp`
- `UV_CACHE_DIR=out/rdna4-ready-missing-profiler-closure-20260613/uv-cache`
- `--subprocess-memory-limit-gib 18`
- `--workload-sharded`
- `--strict-isolation`
- `--gpu-device 0`
- `--timeout 900`

Each target writes to `out/rdna4-ready-missing-profiler-closure-20260613/l2-single-<NN>/`.
After each target, clean only workspace staging dirs matching `tmp/sol_execbench_rdna4_prof_batch_*` and recompute coverage when useful.

## Verification

- Each target has either profiler-backed timing evidence or an explicit `profiler_blocked`/failure summary.
- Recompute coverage using new single-target timing dirs before previous evidence dirs.
- Final accounting reports remaining `ready_missing_profiler_timing`, `profiler_blocked`, `partial_profiler_backed`, and coverage percentage.

## Final Result

- Valid final batch: `out/rdna4-ready-missing-profiler-closure-20260613/l2-batch-03/`
- First target sidecar: `out/rdna4-ready-missing-profiler-closure-20260613/l2-single-00/timing/L2/012_moe_expert_batched_execution_with_capacity_factor.timing.json`
- Final coverage artifact: `out/rdna4-ready-missing-profiler-closure-20260613/coverage-stage2/`
- `ready_missing_profiler_timing_problems`: 29 -> 0
- `profiler_backed_problems`: 121 / 235, unchanged
- `profiler_backed_coverage_pct`: 51.4894%, unchanged
- `profiler_blocked_problems`: 6 -> 35
- `partial_profiler_backed_problems`: 3, unchanged
- `reference_oom_blocked_problems`: 35, unchanged

The quick task closed the remaining L2 ready-missing accounting gap by producing
explicit `profiler_blocked` evidence for all 29 targets. It did not add new
profiler-backed timing because every target blocked under workload-sharded
`rocprofv3` collection or dynamic memory preflight.

`l2-batch-02` is intentionally excluded from the final coverage merge because it
was launched inside the Codex sandbox, where ROCm GPU devices were not visible
to PyTorch. The valid resumed run is `l2-batch-03`, launched with elevated host
GPU access.

Failure reason distribution across the 29 target sidecars:

- 23 problems had at least one workload where `rocprofv3` exited with `-11`.
- 5 problems exceeded the dynamic estimated timing-input cap before launching
  the eval subprocess.
- 1 problem produced no kernel activity rows under `rocprofv3`.
- Across all workload slices: 364 `rocprofv3` exit `-11`, 80 dynamic preflight
  cap blocks, and 16 no-kernel-row blocks.

## OOM Incident And Fix

The first attempted target, `L2/012_moe_expert_batched_execution_with_capacity_factor`,
triggered a system OOM before any workload slice timing sidecar was written.
Kernel logs showed the killed process was a Python eval subprocess with roughly
23.8 GiB anonymous RSS:

```text
Out of memory: Killed process 3823705 (python) total-vm:33895036kB, anon-rss:23787844kB
```

Fix applied:

- Added `--subprocess-memory-limit-gib` to
  `scripts/run_rdna4_profiler_timing_batch.py`.
- The staging runner now applies `RLIMIT_AS` to profiler subprocesses when this
  option is set, so oversized workloads fail inside the child process instead
  of triggering a system-wide OOM kill.
- Added regression coverage for the memory-limited staging runner.
- Verified with:
  `uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py`
  (`36 passed` after the preflight regression was added).

Root cause:

- `L2/012` is not a normal ready-missing target. Its custom input generator
  creates three all-expert BF16 weight tensors:
  `160 * 6144 * 2560 * 2` bytes each, about 4.69 GiB per tensor and about
  14.06 GiB for the three tensors before smaller inputs.
- The benchmark timing path uses `ShiftingMemoryPoolAllocator`, which builds a
  separate timing input pool for every tensor input to keep per-iteration
  `data_ptr` values unique. That is correct for timing isolation, but for this
  MoE workload it means the generated inputs and the timing pool copy coexist.
- Static estimate for `L2/012`:
  - workload 0: input 14.11 GiB, timing input pool peak 28.22 GiB.
  - workload 8: input 14.16 GiB, timing input pool peak 28.31 GiB.
- The system OOM log killed the eval Python process at about 23.8 GiB anonymous
  RSS, which matches the expected growth before all intermediate tensors are
  even accounted for.

Follow-up fix:

- Added `--max-estimated-timing-input-gib` to
  `scripts/run_rdna4_profiler_timing_batch.py`.
- The profiler batch now estimates per-workload tensor input bytes from
  `definition.json` and `workload.jsonl` before staging/running the subprocess.
  It accounts for the generated inputs plus the timing allocator input-pool
  copy.
- Workloads above the cap are classified as `profiler_blocked` before the eval
  subprocess starts, with an explicit failure reason containing the estimated
  input GiB, timing-pool peak GiB, cap, and workload UUID.
- The cap is dynamic by default. When no manual
  `--max-estimated-timing-input-gib` is provided, the batch reads current
  `/proc/meminfo` `MemAvailable`, cgroup remaining memory when present, and
  the optional `--subprocess-memory-limit-gib`; it uses the smallest available
  value times a 70% safety factor. `--no-auto-estimated-timing-input-cap`
  disables this default, and a manual cap still overrides it.
- On the current host at investigation time, dynamic available memory was about
  20.67 GiB, producing a default timing input cap of about 14.47 GiB.
- Verified with:
  `uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py`
  (`37 passed`).

Operational rule after this incident:

- Do not resume L2 ready-missing profiling without
  `--subprocess-memory-limit-gib`.
- Leave the default dynamic timing input cap enabled for the remaining L2
  profiler closure run. Only pass a manual `--max-estimated-timing-input-gib`
  when intentionally overriding the host-derived decision.
- Start with `18` GiB or lower while the desktop session is active; lower to
  `12` GiB for targets that already showed OOM behavior.
- Treat preflight-cap and memory-limit failures as `profiler_blocked` evidence
  rather than retrying unbounded.

## Current Targets

- `L2/012_moe_expert_batched_execution_with_capacity_factor`
- `L2/013_expert_weighted_aggregation_with_shared_expert`
- `L2/014_audio_encoder_varlen_attention_with_chunking_backward`
- `L2/017_fused_vision_cu_seqlens_attention_with_2d_rope_backward`
- `L2/018_cu_seqlens_variable_length_vision_attention`
- `L2/024_moe_expert_parallel_execution`
- `L2/025_moe_expert_parallel_execution_backward`
- `L2/026_moe_expert_parallel_execution_with_weighted_aggregation`
- `L2/027_grouped_query_attention_with_yarn_rope_and_qk_norm`
- `L2/031_flux_timestep_guidance_projection_embedding`
- `L2/032_dual_stream_attention_with_conditional_cross_attention`
- `L2/034_vision_language_cross_attention_fusion`
- `L2/036_convnextv2_layer_with_nhwc_persistence_backward`
- `L2/037_qk_norm_rope_attention_core_backward`
- `L2/038_audio_relative_position_attention`
- `L2/039_kv_shared_attention_with_rope`
- `L2/041_kv_shared_attention_with_dual_rope`
- `L2/042_ffn_gelu_projection_fused_backward`
- `L2/045_audio_encoder_to_language_model_multimodal_fusion`
- `L2/047_moe_training_token_repeat_and_expert_computation`
- `L2/048_moe_expert_inference_batched_dispatch`
- `L2/051_seqlen-finetuned-reconstructed_hyena_complete_forward_block`
- `L2/056_language_model_decoder_prenorm_attention_ffn_residual_backward`
- `L2/065_sparse_expert_dispatch_and_combine`
- `L2/068_gelu_approximate_feedforward_backward`
- `L2/071_edit_consistency_loss_with_perceptual_weighting`
- `L2/072_region_aware_self_attention_with_edit_bias_backward`
- `L2/076_sam_hq_vision_attention_with_relative_position_backward`
- `L2/080_moe_complete_layer_with_shared_expert_backward`
