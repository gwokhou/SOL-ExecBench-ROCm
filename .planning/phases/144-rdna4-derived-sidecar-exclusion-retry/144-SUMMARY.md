---
phase: 144
title: RDNA4 derived sidecar exclusion retry
status: completed
completed: 2026-06-08
---

# Phase 144 Summary

## Result

Phase 144 retried the Phase 140 temporary RDNA4 derived sidecar exclusions
through isolated per-problem `systemd-run --user` jobs. The retry used
`MemoryMax=24G`, `MemorySwapMax=0`, and `UV_CACHE_DIR=/home/guohao/.cache/uv`
so memory-heavy sidecar generation would not remain in the Codex process tree.

The retry covered 10 exclusion records across 8 target problems, representing
the 56 temporary derived sidecar workload exclusions in the Phase 140 evidence
bundle.

## Evidence

- Retry targets:
  `out/rdna4-derived-retry-v131/logs/retry-problem-ids.txt`
- Retry status JSONL:
  `out/rdna4-derived-retry-v131/logs/isolated-derived-status.jsonl`
- Retry log:
  `out/rdna4-derived-retry-v131/logs/isolated-derived.log`
- Machine-readable summary:
  `out/rdna4-derived-retry-v131/phase144-summary.json`
- Human-readable summary:
  `out/rdna4-derived-retry-v131/phase144-summary.md`

## Classification

| Problem | Retry status | systemd result | Memory peak | AMD sidecars | SOLAR sidecars |
|---|---:|---:|---:|---:|---:|
| `L1/060_fused_attention_qk_matmul_scale_mask_softmax_backward` | failed | oom-kill | 24.0G | 14 | 14 |
| `L1/067_flash_attention_gqa_ultralong` | failed | oom-kill | 24.0G | 16 | 16 |
| `L1/076_batched_expert_forward` | failed | oom-kill | 24.0G | 2 | 2 |
| `L2/002_decoder_layer_full_block` | failed | oom-kill | 24.0G | 17 | 17 |
| `L2/053_text_decoder_layer_with_self_attention_and_mlp` | ok | success | 524.0K | 16 | 16 |
| `L2/055_audio_encoder_conv_positional_layer_stack` | failed | oom-kill | 24.0G | 3 | 3 |
| `L2/059_decoder_layer_full_block` | failed | oom-kill | 24.0G | 7 | 7 |
| `L2/070_basic_transformer_block` | failed | oom-kill | 24.0G | 1 | 1 |

The retry generated 76 AMD SOL v2 sidecars and 76 SOLAR derivation sidecars
under the v1.31 retry evidence root. One target problem completed fully:
`L2/053_text_decoder_layer_with_self_attention_and_mlp`. Seven target problems
remain memory blockers under this validation host and the 24G no-swap
per-problem cap.

## Host Boundary

The verification machine is part of the conclusion boundary because these
derived retries are constrained by host memory.

- Hardware model: Gigabyte Technology Co., Ltd. B850 AI TOP
- CPU: AMD Ryzen 9 9950X
- System memory: 32.0 GiB
- GPU under test: AMD Navi 44 / Radeon RX 9060 XT, `rocminfo` agent `gfx1200`,
  16 GiB VRAM
- Other GPU present: NVIDIA GeForce RTX 5060 Ti, not used for RDNA4 validation
- OS: Ubuntu 24.04.4 LTS, Linux 6.17.0-35-generic

## Conclusion

Phase 144 closes the temporary-exclusion retry requirement by converting the
Phase 140 sidecar gaps into auditable recovered/blocker classifications. The
remaining blocker is not a Codex stability issue: OOM was contained to
transient systemd child units, while Codex and the calling shell survived.

