---
status: complete
trigger: "排查剩余的 10 个 RDNA4 partial_profiler_backed profiler timing 问题"
created: "2026-06-08T22:55:42+08:00"
updated: "2026-06-08T23:00:00+08:00"
outcome: diagnosed
---

# RDNA4 Partial Profiler Targets Debug

## Symptoms

- Expected behavior: remaining partial profiler-backed targets should either be
  closed with full rocprofv3-backed timing or explained with actionable blockers.
- Actual behavior: 10 targets remain `partial_profiler_backed`.
- Known evidence: `out/rdna4-profiler-partial-failure-classification-20260608/`.
- Reproduction: inspect timing sidecars for the 10 partial targets and correlate
  failing workloads with derived AMD solution artifacts.

## Current Focus

- hypothesis: The remaining 10 targets are blocked by workload-specific
  reference/gen_inputs OOM on the current 16GB RDNA4 device, not by profiler
  lifetime or CSV parsing.
- test: Extract failing workload indexes, trace status details, and candidate
  solution provenance from timing sidecars and derived artifacts.
- expecting: Each target maps to one of correctness, runtime, or mixed blockers
  with explicit failing workload evidence.
- next_action: choose whether to exclude oversized reference workloads from the
  RDNA4 denominator, add a memory-footprint readiness classifier, or validate on
  a larger-memory AMD GPU before claiming full coverage

## Evidence

- 2026-06-08T22:55:42+08:00: Phase 161 classification lists 10 partial targets:
  6 correctness-only, 2 runtime-only, and 2 mixed correctness/runtime.
- 2026-06-08T22:58:17+08:00: Isolated workload-slice rerun for
  `L1/013_fused_residual_rms_norm_backward` offset 1 still produced
  `INVALID_REFERENCE`; artifact:
  `out/rdna4-profiler-debug-l1013-offset1-20260608/`.
- 2026-06-08T22:59:01+08:00: Same isolated workload with
  `PYTORCH_ALLOC_CONF=expandable_segments:True` still produced
  `INVALID_REFERENCE`; ROCm/PyTorch warned expandable segments are unsupported
  on this HIP platform. Artifact:
  `out/rdna4-profiler-debug-l1013-offset1-expandable-20260608/`.
- 2026-06-08T23:00:00+08:00: Timing sidecar JSONL traces show all non-PASSED
  statuses are HIP OOMs in `gen_inputs`, reference `run()`, or the user function
  path when the user function is the staged reference solution.

## Diagnosis

The remaining 10 `partial_profiler_backed` targets are not closed by retrying
rocprofv3. The profiler collected evidence for many attempted workloads, but
the staged reference timing path cannot produce `PASSED` traces for oversized
workloads on this 16GB RDNA4 device.

Per-target root cause:

- `L1/002_vae_conv3x3_groupnorm_silu_residual_fused`: offset 10,
  `batch_size=1,height=1024,width=1024`, reference conv/groupnorm path OOM.
- `L1/013_fused_residual_rms_norm_backward`: offset 1,
  `batch_size=64,seq_len=8192`, reference RMSNorm backward OOM even isolated.
- `L1/015_grouped_query_attention_with_rope_and_qk_norm`: offset 6,
  `batch_size=1,seq_len=8192`, dense attention/GQA reference OOM.
- `L1/047_attention_with_qk_norm_and_rope`: offset 10,
  `batch_size=1,seq_len=8192`, dense attention reference OOM.
- `L1/067_flash_attention_gqa_ultralong`: offsets 16-17,
  `seq_len=8192/16384`, dense attention reference OOM.
- `L1/070_mamba2_fused_intra_chunk_diagonal_computation`: offset 9,
  `batch_size=4,num_chunks=16`, broadcasted Mamba diagonal tensors OOM.
- `L1/026_video_patch_embedding_projection`: offsets 0-2 fail with
  reference/user-function OOM and offset 3 remains profiler-blocked; sharding
  avoided process lifetime failure but did not make the reference path pass.
- `L2/005_swiglu_mlp_backward`: all 18 workloads fail with reference or staged
  user-function OOM; this is a full-problem memory-footprint blocker.
- `L1/060_fused_attention_qk_matmul_scale_mask_softmax_backward`: offset 14,
  `seq_len_q=4096,seq_len_k=4096`, input generation/attention tensor OOM.
- `L1/076_batched_expert_forward`: all 14 workloads fail in input generation
  with a repeated 7.91 GiB allocation pattern before timing can pass.

## Recommended Follow-Up

- Add a memory-footprint/readiness classifier for RDNA4 profiler timing that
  separates `profiler_blocked` from `reference_oom_blocked`.
- Do not count these 10 as profiler coverage until a passing reference trace is
  available.
- If full 235-problem coverage is still required, validate these oversized
  reference workloads on a larger-memory AMD GPU or introduce a documented
  RDNA4 memory-denominator policy.

## Eliminated

- hypothesis: The 10 targets are all profiler lifecycle/OOM failures.
  reason: Phase 161 ledger shows non-PASSED workload statuses even when
  rocprofv3 produced kernel rows for many or all attempted workloads.
