# Phase 138 Run Checkpoint

## 2026-06-07T15:55:38Z

### Preconditions

- SOL-ExecBench benchmark data exists under `data/SOL-ExecBench/benchmark`.
- Dataset layout verification manifest written to
  `out/rdna4-full-dataset/sol-dataset-manifest.json`.
- Inventory/readiness/ready-subset sidecars written to:
  - `out/rdna4-full-dataset/inventory.json`
  - `out/rdna4-full-dataset/readiness.json`
  - `out/rdna4-full-dataset/ready_subset.json`
- Ready subset denominator:
  - total workloads: 3957
  - ready/included workloads: 1907
  - excluded/readiness-blocked workloads: 2050
  - ready categories: L1 and L2
- Escalated `uv run` ROCm probe sees:
  - `/dev/kfd True`
  - `/dev/dri True`
  - HIP `7.1.25424`
  - `torch.cuda.is_available() True`
  - device count `1`
  - arch `gfx1200`

### Background Tasks

- `hf download flashinfer-ai/flashinfer-trace` is still running from
  `scripts/download_data.sh`. It is not required for the current ready-subset
  run because FlashInfer workloads are readiness-blocked, but it is being
  allowed to continue.

### Dataset Run

Started with:

```bash
UV_CACHE_DIR=/home/guohao/.cache/uv uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --output out/rdna4-full-dataset/run --ready-subset out/rdna4-full-dataset/ready_subset.json --readiness out/rdna4-full-dataset/readiness.json --dataset-manifest out/rdna4-full-dataset/sol-dataset-manifest.json --execution-closure out/rdna4-full-dataset/execution_closure.json --gpu-architecture gfx1200 --timeout-policy record --workload-shard-size 1 --rerun
```

Codex session id: `9739`.

Observed progress:

- Current active problem: `L1/002_vae_conv3x3_groupnorm_silu_residual_fused`
- Current active shard at latest poll: `workload_shard_0011.jsonl`
- No `traces.json` written yet because the active problem has not completed.

### Policy

Do not terminate healthy long-running validation processes solely due to
elapsed time. Poll and preserve partial artifacts.

## 2026-06-07T16:05:59Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- `UV_CACHE_DIR` is set to `/home/guohao/.cache/uv` for the long-running
  invocation.
- First completed problem:
  `L1/002_vae_conv3x3_groupnorm_silu_residual_fused`.
- First completed problem output:
  `out/rdna4-full-dataset/run/L1/002_vae_conv3x3_groupnorm_silu_residual_fused/traces.json`.
- First completed problem workload status distribution:
  - `PASSED`: 19
  - `INVALID_REFERENCE`: 1
- The `INVALID_REFERENCE` record is a real RDNA4 execution finding caused by
  HIP out-of-memory in the PyTorch reference implementation on a 15.92 GiB
  RDNA4 device.
- Latest active problem at this checkpoint:
  `L1/004_attention_output_projection_with_reshape_backward`.
- Latest active shard at this checkpoint: `workload_shard_0015.jsonl`.

### Background Tasks

- `hf download flashinfer-ai/flashinfer-trace` remains active and is being
  allowed to continue. The current Phase 138 ready subset does not depend on it
  because FlashInfer workloads are readiness-blocked in the sidecar output.

## 2026-06-07T16:13:53Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- Completed problem count: 11.
- Completed workload count: 179.
- Completed workload status distribution:
  - `PASSED`: 176
  - `INVALID_REFERENCE`: 3
- All observed `INVALID_REFERENCE` records are HIP out-of-memory failures in
  the PyTorch reference path on the 15.92 GiB RDNA4 device:
  - `L1/002_vae_conv3x3_groupnorm_silu_residual_fused`
  - `L1/013_fused_residual_rms_norm_backward`
  - `L1/015_grouped_query_attention_with_rope_and_qk_norm`
- Latest completed problem at this checkpoint:
  `L1/016_rope_inverse_frequency_computation`.
- Latest active problem at this checkpoint:
  `L1/019_vision_3d_rotary_embedding_with_spatial_merge_indexing_backward`.
- Latest active shard at this checkpoint: `workload_shard_0004.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 1.3 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 1.2 GiB and the background download
  remains active.

## 2026-06-07T16:25:07Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- Main runner stdout has flushed and confirms it discovered 121 ready problems
  under `data/SOL-ExecBench/benchmark`.
- Completed/trace-bearing problem count: 21.
- Completed/trace-bearing workload count: 317.
- Observed workload status distribution:
  - `PASSED`: 308
  - `INVALID_REFERENCE`: 8
  - `RUNTIME_ERROR`: 1
- The observed `RUNTIME_ERROR` is also a HIP out-of-memory condition in the
  user-function path for
  `L1/026_video_patch_embedding_projection`; it is being preserved as a real
  RDNA4 capacity-bound execution finding.
- Latest completed problem at this checkpoint: `L1/033_post_norm_residual`.
- Latest active problem at this checkpoint:
  `L1/034_flux_multi_axis_rope_frequency_computation`.
- Latest active shard at this checkpoint: `workload_shard_0001.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 2.3 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 2.0 GiB and the background download
  remains active.

## 2026-06-07T16:42:00Z

### Long-Tail Observation

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- `L1/037_flux_feedforward_gelu_approximate` is a newly observed long-tail
  candidate.
- At the latest poll, the active shard was `workload_shard_0007.jsonl`.
- The active `eval_driver.py` process had exceeded five minutes of wall time
  while still using the RDNA4 GPU at full utilization.
- This indicates `--timeout 300` is not acting as an outer hard-kill boundary
  for this execution path. The process is being allowed to continue because it
  is healthy and still producing GPU activity.
- This problem should be reviewed for possible long-tail exclusion treatment
  after the run reaches a natural checkpoint or completion.

## 2026-06-07T17:06:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- The polling cadence was reduced to approximately 10 minutes for long-running
  background validation tasks.
- Completed/trace-bearing problem count: 33.
- Completed/trace-bearing workload count: 507.
- Observed workload status distribution:
  - `PASSED`: 495
  - `INVALID_REFERENCE`: 9
  - `RUNTIME_ERROR`: 1
  - `TIMEOUT`: 2
- `L1/037_flux_feedforward_gelu_approximate` reached a natural problem-level
  checkpoint and contributed the first observed `TIMEOUT` statuses.
- Latest completed problem at this checkpoint:
  `L1/047_attention_with_qk_norm_and_rope`.
- Latest active problem at this checkpoint:
  `L1/048_fused_gate_up_projection_with_swiglu`.
- Latest active shard at this checkpoint: `workload_shard_0004.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 3.6 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 10.2 GiB and the background download
  remains active.

## 2026-06-07T17:28:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- Completed/trace-bearing problem count: 49.
- Completed/trace-bearing workload count: 757.
- Observed workload status distribution:
  - `PASSED`: 743
  - `INVALID_REFERENCE`: 9
  - `RUNTIME_ERROR`: 3
  - `TIMEOUT`: 2
- Main runner stdout confirms successful progression through the 40-problem
  boundary and records `L1/037_flux_feedforward_gelu_approximate` as 14/16
  passed with two `TIMEOUT` statuses.
- Latest completed problem at this checkpoint:
  `L1/066_masked_softmax_with_attention_dropout_backward`.
- Latest active problem at this checkpoint:
  `L1/067_flash_attention_gqa_ultralong`.
- Latest active shard at this checkpoint: `workload_shard_0009.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 5.3 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T17:39:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- Completed/trace-bearing problem count: 58.
- Completed/trace-bearing workload count: 897.
- Observed workload status distribution:
  - `PASSED`: 864
  - `INVALID_REFERENCE`: 14
  - `RUNTIME_ERROR`: 17
  - `TIMEOUT`: 2
- Main runner stdout confirms progression past `[57/121]` and records several
  high-memory failures, including:
  - `L1/060_fused_attention_qk_matmul_scale_mask_softmax_backward`
  - `L1/067_flash_attention_gqa_ultralong`
  - `L1/070_mamba2_fused_intra_chunk_diagonal_computation`
  - `L1/076_batched_expert_forward`
- The `RUNTIME_ERROR` count increased materially in this interval and should be
  classified during closure analysis against OOM, generation/input setup, and
  timing-path failures.
- Latest completed problem at this checkpoint:
  `L1/077_whisper_decoder_output_projection`.
- Latest active problem at this checkpoint: `L1/078_group_norm_fusion`.
- Latest active shard at this checkpoint: `workload_shard_0013.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 6.3 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T18:21:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- The run has crossed from L1 into L2 workloads.
- Completed/trace-bearing problem count: 73.
- Completed/trace-bearing workload count: 1137.
- Observed workload status distribution:
  - `PASSED`: 1104
  - `INVALID_REFERENCE`: 14
  - `RUNTIME_ERROR`: 17
  - `TIMEOUT`: 2
- Latest completed problem at this checkpoint:
  `L2/001_fused_vision_multihead_attention_with_norms_backward`.
- Latest active problem at this checkpoint: `L2/002_decoder_layer_full_block`.
- Latest active shard at this checkpoint: `workload_shard_0002.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 8.1 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T19:03:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- `L2/002_decoder_layer_full_block` reached a natural checkpoint after a long
  interval.
- Completed/trace-bearing problem count: 75.
- Completed/trace-bearing workload count: 1171.
- Observed workload status distribution:
  - `PASSED`: 1133
  - `INVALID_REFERENCE`: 16
  - `RUNTIME_ERROR`: 17
  - `TIMEOUT`: 5
- The timeout count increased from 2 to 5 during this interval, consistent with
  L2 long-tail execution behavior.
- Latest completed problem at this checkpoint:
  `L2/003_grouped_query_attention_with_rope_backward`.
- Latest active problem at this checkpoint: `L2/004_fused_residual_rms_mlp`.
- Latest active shard at this checkpoint: `workload_shard_0014.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 8.3 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T19:14:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- Completed/trace-bearing problem count: 79.
- Completed/trace-bearing workload count: 1237.
- Observed workload status distribution:
  - `PASSED`: 1179
  - `INVALID_REFERENCE`: 34
  - `RUNTIME_ERROR`: 19
  - `TIMEOUT`: 5
- Main runner stdout confirms progression through `[77/121]`; `L2/004` added
  OOM-class `RUNTIME_ERROR` records, and later L2 workloads increased
  `INVALID_REFERENCE` materially.
- Latest completed problem at this checkpoint:
  `L2/008_moe_sparse_routing_and_dispatch`.
- Latest active problem at this checkpoint:
  `L2/015_audio_sinusoidal_position_embedding_with_conv_projection`.
- Latest active shard at this checkpoint: `workload_shard_0003.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 8.9 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T20:07:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- `L2/019_decoder_layer_fused_attention_mlp` reached a natural checkpoint after
  a long interval.
- Completed/trace-bearing problem count: 84.
- Completed/trace-bearing workload count: 1317.
- Observed workload status distribution:
  - `PASSED`: 1256
  - `INVALID_REFERENCE`: 35
  - `RUNTIME_ERROR`: 19
  - `TIMEOUT`: 7
- Latest completed problem at this checkpoint:
  `L2/021_cross_attention_text_video_conditioning_backward`.
- Latest active problem at this checkpoint:
  `L2/022_video_latent_denoising_unet_block`.
- Latest active shard at this checkpoint: `workload_shard_0003.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 9.5 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T21:36:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- `L2/023_video_latent_vae_encoder_downsampling` reached a natural checkpoint
  after a long interval.
- Completed/trace-bearing problem count: 86.
- Completed/trace-bearing workload count: 1349.
- Observed workload status distribution:
  - `PASSED`: 1272
  - `INVALID_REFERENCE`: 37
  - `RUNTIME_ERROR`: 19
  - `TIMEOUT`: 21
- The timeout count increased from 7 to 21 during this interval, making
  `L2/023_video_latent_vae_encoder_downsampling` a high-priority long-tail
  exclusion-review candidate for denominator closure.
- Latest completed problem at this checkpoint:
  `L2/023_video_latent_vae_encoder_downsampling`.
- Latest active problem at this checkpoint:
  `L2/028_gqa_rotary_attention_core_backward`.
- Latest active shard at this checkpoint: `workload_shard_0014.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 9.9 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T21:47:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- Completed/trace-bearing problem count: 92.
- Completed/trace-bearing workload count: 1438.
- Observed workload status distribution:
  - `PASSED`: 1349
  - `INVALID_REFERENCE`: 39
  - `RUNTIME_ERROR`: 28
  - `TIMEOUT`: 21
  - `INCORRECT_NUMERICAL`: 1
- This interval introduced the first observed `INCORRECT_NUMERICAL` status and
  should be classified during closure analysis separately from OOM and timeout
  failures.
- Main runner stdout confirms `L2/029_moe_sparse_routing_and_dispatch` as
  0/9 passed with OOM-class `RUNTIME_ERROR` records.
- Latest completed problem at this checkpoint:
  `L2/040_altup_predict_correction_cycle_backward`.
- Latest active problem at this checkpoint:
  `L2/043_mamba_chunk_scan_with_segsum`.
- Latest active shard at this checkpoint: `workload_shard_0009.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 10.6 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T22:08:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- Completed/trace-bearing problem count: 100.
- Completed/trace-bearing workload count: 1561.
- Observed workload status distribution:
  - `PASSED`: 1471
  - `INVALID_REFERENCE`: 40
  - `RUNTIME_ERROR`: 28
  - `TIMEOUT`: 21
  - `INCORRECT_NUMERICAL`: 1
- The run has crossed the 100-problem boundary with no new status categories
  since the first `INCORRECT_NUMERICAL` observation.
- Latest completed problem at this checkpoint:
  `L2/054_vision_encoder_layer_with_gated_residuals`.
- Latest active problem at this checkpoint:
  `L2/055_audio_encoder_conv_positional_layer_stack`.
- Latest active shard at this checkpoint: `workload_shard_0003.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 11.6 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T23:01:55Z

### Long-Tail Observation

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- `L2/055_audio_encoder_conv_positional_layer_stack` is a newly observed
  long-tail candidate.
- The problem has remained active across multiple 10-minute polling intervals
  without writing a problem-level `traces.json` yet.
- Latest active shard at this checkpoint: `workload_shard_0014.jsonl`.
- The active `eval_driver.py` process remains CPU/GPU active and is being
  allowed to continue under the long-running validation policy.
- Completed/trace-bearing problem count remains 100.
- Completed/trace-bearing workload count remains 1561.
- Observed workload status distribution remains:
  - `PASSED`: 1471
  - `INVALID_REFERENCE`: 40
  - `RUNTIME_ERROR`: 28
  - `TIMEOUT`: 21
  - `INCORRECT_NUMERICAL`: 1
- `out/rdna4-full-dataset/run` is approximately 11.6 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T23:13:00Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- `L2/055_audio_encoder_conv_positional_layer_stack` reached a natural
  problem-level checkpoint after a long interval.
- Completed/trace-bearing problem count: 102.
- Completed/trace-bearing workload count: 1592.
- Observed workload status distribution:
  - `PASSED`: 1493
  - `INVALID_REFERENCE`: 40
  - `RUNTIME_ERROR`: 28
  - `TIMEOUT`: 30
  - `INCORRECT_NUMERICAL`: 1
- `L2/055_audio_encoder_conv_positional_layer_stack` contributed 9 additional
  `TIMEOUT` statuses and is a high-priority long-tail exclusion-review
  candidate.
- `L2/057_residual_coupling_flow_block` completed 16/16 passed.
- Latest completed problem at this checkpoint:
  `L2/057_residual_coupling_flow_block`.
- Latest active problem at this checkpoint: `L2/058_mamba2_selective_scan`.
- Latest active shard at this checkpoint: `workload_shard_0006.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 11.9 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-07T23:45:14Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- The run has crossed the 110-problem boundary.
- Completed/trace-bearing problem count: 110.
- Completed/trace-bearing workload count: 1720.
- Observed workload status distribution:
  - `PASSED`: 1617
  - `INVALID_REFERENCE`: 44
  - `RUNTIME_ERROR`: 28
  - `TIMEOUT`: 30
  - `INCORRECT_NUMERICAL`: 1
- No new status categories were introduced since the prior checkpoint.
- Recently completed problems `L2/062_decoder_complete_layer`,
  `L2/063_encoder_layer_dual_residual_norm_chain`,
  `L2/064_multi_head_qkv_projection_with_rope_backward`, and
  `L2/066_resnet_block_with_time_embedding` each completed 16/16 passed.
- Latest completed problem at this checkpoint:
  `L2/066_resnet_block_with_time_embedding`.
- Latest active problem at this checkpoint:
  `L2/067_patch_embed_to_joint_attention_input`.
- Latest active shard at this checkpoint: `workload_shard_0008.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 12.9 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-08T00:06:44Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- Completed/trace-bearing problem count: 112.
- Completed/trace-bearing workload count: 1751.
- Observed workload status distribution:
  - `PASSED`: 1647
  - `INVALID_REFERENCE`: 44
  - `RUNTIME_ERROR`: 29
  - `TIMEOUT`: 30
  - `INCORRECT_NUMERICAL`: 1
- `L2/069_joint_transformer_block_residual_path` completed 15/16 passed and
  contributed one additional `RUNTIME_ERROR`.
- The `L2/069` runtime error is a HIP out-of-memory failure in the user
  function path for `seq_len=8192`, with a failed 6.12 GiB allocation on the
  15.92 GiB RDNA4 device.
- Latest completed problem at this checkpoint:
  `L2/069_joint_transformer_block_residual_path`.
- Latest active problem at this checkpoint: `L2/070_basic_transformer_block`.
- Latest active shard at this checkpoint: `workload_shard_0015.jsonl`.
- `out/rdna4-full-dataset/run` is approximately 13.2 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-08T00:38:30Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- Completed/trace-bearing problem count: 116.
- Completed/trace-bearing workload count: 1815.
- Observed workload status distribution:
  - `PASSED`: 1707
  - `INVALID_REFERENCE`: 47
  - `RUNTIME_ERROR`: 29
  - `TIMEOUT`: 31
  - `INCORRECT_NUMERICAL`: 1
- `L2/073_feedforward_mlp_backward` completed 15/16 passed and contributed one
  additional `TIMEOUT`.
- `L2/074_sam_hq_mask_decoder_iou_hypernetwork_fusion_backward` and
  `L2/075_sam_hq_mask_decoder_two_way_transformer` each completed 16/16 passed.
- Latest completed problem at this checkpoint:
  `L2/075_sam_hq_mask_decoder_two_way_transformer`.
- Latest active problem at this checkpoint:
  `L2/077_sam_hq_vision_encoder_window_partition_attention`.
- Latest active shard at this checkpoint: `workload_shard_0008.jsonl`.
- Remaining ready problems after this checkpoint: 5.
- `out/rdna4-full-dataset/run` is approximately 13.8 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.

## 2026-06-08T01:00:11Z

### Progress Update

- The escalated RDNA4 dataset run is still active in Codex session `9739`.
- Completed/trace-bearing problem count: 117.
- Completed/trace-bearing workload count: 1831.
- Observed workload status distribution:
  - `PASSED`: 1717
  - `INVALID_REFERENCE`: 52
  - `RUNTIME_ERROR`: 30
  - `TIMEOUT`: 31
  - `INCORRECT_NUMERICAL`: 1
- `L2/077_sam_hq_vision_encoder_window_partition_attention` completed 10/16
  passed, with five `INVALID_REFERENCE` records and one `RUNTIME_ERROR`.
- All non-passing `L2/077` records sampled from the trace/log are HIP
  out-of-memory failures, split between the reference path and the user
  function path.
- Latest completed problem at this checkpoint:
  `L2/077_sam_hq_vision_encoder_window_partition_attention`.
- Latest active problem at this checkpoint:
  `L2/078_fused_final_layer_upsample_with_adaptive_norm`.
- Latest active shard at this checkpoint: `workload_shard_0007.jsonl`.
- Remaining ready problems after this checkpoint: 4.
- `out/rdna4-full-dataset/run` is approximately 14.0 MiB at this checkpoint.
- `data/flashinfer-trace` is approximately 14.7 GiB and the background download
  remains active.
