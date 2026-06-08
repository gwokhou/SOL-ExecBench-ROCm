# Phase 138 Summary

## Result

Phase 138 completed the RDNA4 `gfx1200` bounded ready-subset dataset execution
and denominator closure on 2026-06-08.

This is not full paper parity, leaderboard authority, upstream SOLAR
equivalence, or full 235-problem validation authority. The completed claim is a
bounded RDNA4 ready-subset execution with complete accounting for all selected
workloads and all readiness-blocked workloads.

## Artifacts

- Dataset manifest:
  `out/rdna4-full-dataset/sol-dataset-manifest.json`
- Inventory:
  `out/rdna4-full-dataset/inventory.json`
- Readiness report:
  `out/rdna4-full-dataset/readiness.json`
- Ready subset:
  `out/rdna4-full-dataset/ready_subset.json`
- Run output:
  `out/rdna4-full-dataset/run`
- Problem summary:
  `out/rdna4-full-dataset/run/summary.json`
- Execution closure:
  `out/rdna4-full-dataset/execution_closure.json`
- Long-running progress log:
  `138-RUN-CHECKPOINT.md`

## Denominator

- Total workload records: 3957
- Ready/included workloads attempted on RDNA4: 1907
- Readiness-blocked workloads not attempted: 2050
  - FlashInfer runtime assumption: 660
  - Custom input missing evaluator evidence: 872
  - NVIDIA/CUDA runtime dependency: 518
- Long-tail exclusions supplied: 0
- Filtered workloads: 0

## RDNA4 Execution Outcome

Problem-level summary:

- Total ready problems: 121
- OK problems: 86
- FAIL problems: 35
- L1: 72 problems, 57 OK, 15 FAIL
- L2: 49 problems, 29 OK, 20 FAIL

Workload-level summary from `summary.json`:

- Total attempted workloads: 1907
- Passed workloads: 1761
- Failed workloads: 146

Closure-level status from `execution_closure.json`:

- `attempted_passed`: 1761
- `attempted_failed`: 134
- `missing_trace`: 12
- `not_attempted`: 2050
- `excluded_long_tail`: 0
- `filtered`: 0
- `derived_evidence_missing`: 0

Trace status distribution:

- `PASSED`: 1761
- `INVALID_REFERENCE`: 52
- `RUNTIME_ERROR`: 46
- `TIMEOUT`: 35
- `INCORRECT_NUMERICAL`: 1

## Notable Findings

- Most `INVALID_REFERENCE` and sampled `RUNTIME_ERROR` records are RDNA4
  capacity-bound HIP out-of-memory findings in either the PyTorch reference
  path, user function path, or input generation path on a 15.92 GiB RDNA4
  device.
- Long-tail timeout candidates observed during the run include:
  - `L1/037_flux_feedforward_gelu_approximate`
  - `L2/002_decoder_layer_full_block`
  - `L2/023_video_latent_vae_encoder_downsampling`
  - `L2/055_audio_encoder_conv_positional_layer_stack`
  - `L2/073_feedforward_mlp_backward`
  - `L2/078_fused_final_layer_upsample_with_adaptive_norm`
- `L2/082_moe_layer_complete_forward_with_residual` completed 0/16 passed; all
  16 failed records are `RUNTIME_ERROR` from HIP out-of-memory during input
  generation.
- The first and only observed `INCORRECT_NUMERICAL` status came from
  `L2/035_convnextv2_block_with_grn`.
- Twelve failed workloads produced no per-workload trace and are explicitly
  represented as `missing_trace` in closure, preventing silent omission.

## Missing Trace Workloads

The 12 `missing_trace` closure entries are distributed across:

- `L1/013_fused_residual_rms_norm_backward`: 1
- `L1/026_video_patch_embedding_projection`: 1
- `L1/028_hybrid_attention_mask_preparation`: 3
- `L1/031_repeat_kv_attention_matmul`: 2
- `L1/053_gaussian_topk_sparse_activation`: 1
- `L1/066_masked_softmax_with_attention_dropout_backward`: 1
- `L1/077_whisper_decoder_output_projection`: 2
- `L2/067_patch_embed_to_joint_attention_input`: 1

## Follow-Up Inputs

- Phase 139 should use the completed traces and closure as the RDNA4 timing
  denominator, while preserving the non-authoritative timing boundary until
  clock and profiler evidence are captured.
- Phase 140 should generate derived reports from the `execution_closure.json`
  and `summary.json` artifacts, with `missing_trace`, OOM, timeout, and
  numerical statuses visible.
- Phase 141 should only upgrade public wording to the bounded evidence level
  supported here.
