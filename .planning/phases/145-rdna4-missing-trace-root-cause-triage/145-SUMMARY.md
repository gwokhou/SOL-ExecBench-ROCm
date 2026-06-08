---
phase: 145
title: RDNA4 missing trace root-cause triage
status: completed
completed: 2026-06-08
---

# Phase 145 Summary

## Result

Phase 145 classified all 12 Phase 138 `missing_trace` workload records. The
records span 8 problems and all resolve to the same root-cause class:
`gpu_oom_no_trace`.

The closure status was correct: no matching workload UUID exists in the merged
`traces.json` file for any of the 12 rows. The missing context was the per-row
root cause. Each row has a corresponding shard CLI log ending in
`torch.OutOfMemoryError: HIP out of memory`.

## Evidence

- Machine-readable triage:
  `out/rdna4-missing-trace-triage-v131/phase145-missing-trace-triage.json`
- Human-readable triage:
  `out/rdna4-missing-trace-triage-v131/phase145-missing-trace-triage.md`
- Source closure:
  `out/rdna4-full-dataset/run/execution_closure.json`

## Classification

| Problem | Missing traces | Classification |
|---|---:|---|
| `L1/013_fused_residual_rms_norm_backward` | 1 | `gpu_oom_no_trace` |
| `L1/026_video_patch_embedding_projection` | 1 | `gpu_oom_no_trace` |
| `L1/028_hybrid_attention_mask_preparation` | 3 | `gpu_oom_no_trace` |
| `L1/031_repeat_kv_attention_matmul` | 2 | `gpu_oom_no_trace` |
| `L1/053_gaussian_topk_sparse_activation` | 1 | `gpu_oom_no_trace` |
| `L1/066_masked_softmax_with_attention_dropout_backward` | 1 | `gpu_oom_no_trace` |
| `L1/077_whisper_decoder_output_projection` | 2 | `gpu_oom_no_trace` |
| `L2/067_patch_embed_to_joint_attention_input` | 1 | `gpu_oom_no_trace` |

The attempted allocations ranged from 810 MiB to 7.58 GiB, while the recorded
free GPU memory at failure ranged from 54 MiB to 7.15 GiB on the 16 GiB
`gfx1200` device.

## Conclusion

The 12 records should remain counted as failed RDNA4 workloads, but they are no
longer unexplained missing artifacts. They are GPU-memory-capacity failures
where the validation shard terminated before emitting a per-workload trace.

