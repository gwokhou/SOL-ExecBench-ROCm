# Quick Task 260613-prof Summary

## Result

Completed. The 24 RDNA4 L2 targets that were not memory-cap blocked no longer
hit profiler-level blockers when collected with kernel trace only.

Batch output:

- Output dir: `out/rdna4-profiler-blocker-fix-20260613/no-hip-runtime-24`
- Replacement timing dir:
  `out/rdna4-profiler-blocker-fix-20260613/no-hip-runtime-24/timing`
- Selected targets: `24`
- `profiler_backed`: `8`
- `partial_profiler_backed`: `16`
- `profiler_blocked`: `0`
- Failed: `0`
- Fallback or missing: `0`

## Implemented Change

Added a profiler batch option to disable rocprofv3 HIP runtime API tracing while
keeping kernel dispatch tracing enabled:

```text
--hip-runtime-trace / --no-hip-runtime-trace
```

This is wired through `run_batch`, workload-sharded execution, and
`Rocprofv3CollectionRequest(include_hip_runtime=...)`.

## Evidence

The successful 24-target batch was run with:

```text
--workload-sharded
--timeout 900
--subprocess-memory-limit-gib 18
--gpu-device 0
--no-hip-runtime-trace
--no-compact-workload-slices
```

Per-target status:

| Target | Status | Workloads | Trace counts |
| --- | --- | ---: | --- |
| `L2/013_expert_weighted_aggregation_with_shared_expert` | `partial_profiler_backed` | `0/16` | `RUNTIME_ERROR=16` |
| `L2/014_audio_encoder_varlen_attention_with_chunking_backward` | `profiler_backed` | `16/16` | `PASSED=16` |
| `L2/017_fused_vision_cu_seqlens_attention_with_2d_rope_backward` | `profiler_backed` | `16/16` | `PASSED=16` |
| `L2/018_cu_seqlens_variable_length_vision_attention` | `profiler_backed` | `16/16` | `PASSED=16` |
| `L2/027_grouped_query_attention_with_yarn_rope_and_qk_norm` | `partial_profiler_backed` | `15/16` | `PASSED=15, INVALID_REFERENCE=1` |
| `L2/031_flux_timestep_guidance_projection_embedding` | `profiler_backed` | `16/16` | `PASSED=16` |
| `L2/032_dual_stream_attention_with_conditional_cross_attention` | `profiler_backed` | `16/16` | `PASSED=16` |
| `L2/034_vision_language_cross_attention_fusion` | `partial_profiler_backed` | `13/16` | `PASSED=13, INVALID_REFERENCE=2` |
| `L2/036_convnextv2_layer_with_nhwc_persistence_backward` | `partial_profiler_backed` | `13/14` | `PASSED=13, RUNTIME_ERROR=1` |
| `L2/037_qk_norm_rope_attention_core_backward` | `partial_profiler_backed` | `14/16` | `PASSED=14, INVALID_REFERENCE=2` |
| `L2/038_audio_relative_position_attention` | `partial_profiler_backed` | `12/14` | `PASSED=12, INVALID_REFERENCE=1, RUNTIME_ERROR=1` |
| `L2/039_kv_shared_attention_with_rope` | `partial_profiler_backed` | `15/16` | `PASSED=15, RUNTIME_ERROR=1` |
| `L2/041_kv_shared_attention_with_dual_rope` | `partial_profiler_backed` | `15/16` | `PASSED=15, INVALID_REFERENCE=1` |
| `L2/042_ffn_gelu_projection_fused_backward` | `partial_profiler_backed` | `14/16` | `PASSED=14, INVALID_REFERENCE=1, RUNTIME_ERROR=1` |
| `L2/045_audio_encoder_to_language_model_multimodal_fusion` | `partial_profiler_backed` | `12/16` | `PASSED=12, INVALID_REFERENCE=1, RUNTIME_ERROR=2` |
| `L2/048_moe_expert_inference_batched_dispatch` | `partial_profiler_backed` | `0/16` | `RUNTIME_ERROR=16` |
| `L2/051_seqlen-finetuned-reconstructed_hyena_complete_forward_block` | `profiler_backed` | `16/16` | `PASSED=16` |
| `L2/056_language_model_decoder_prenorm_attention_ffn_residual_backward` | `partial_profiler_backed` | `8/16` | `PASSED=8, INVALID_REFERENCE=2, RUNTIME_ERROR=1` |
| `L2/065_sparse_expert_dispatch_and_combine` | `partial_profiler_backed` | `0/16` | `RUNTIME_ERROR=16` |
| `L2/068_gelu_approximate_feedforward_backward` | `profiler_backed` | `16/16` | `PASSED=16` |
| `L2/071_edit_consistency_loss_with_perceptual_weighting` | `partial_profiler_backed` | `15/16` | `PASSED=15, INVALID_REFERENCE=1` |
| `L2/072_region_aware_self_attention_with_edit_bias_backward` | `partial_profiler_backed` | `14/16` | `PASSED=14, RUNTIME_ERROR=2` |
| `L2/076_sam_hq_vision_attention_with_relative_position_backward` | `partial_profiler_backed` | `15/16` | `PASSED=15, INVALID_REFERENCE=1` |
| `L2/080_moe_complete_layer_with_shared_expert_backward` | `profiler_backed` | `16/16` | `PASSED=16` |

## Root Cause Boundary

The original `rocprofv3 exit -11` failures are attributable to rocprofiler-sdk
HIP runtime API tracing, not kernel dispatch tracing. Source-level upstream
location is:

- `projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hip/hip.cpp`
- function template:
  `rocprofiler::hip::hip_api_impl<TableIdx, OpIdx>::functor<RetT, Args...>`

This wrapper is entered when HIP runtime tracing is enabled and PyTorch issues
HIP runtime calls during input generation / kernel launch. The crash was not
stable enough to reproduce under gdb after the batch completed, so the precise
line-level `addr2line` mapping from the old PC is unavailable. Function-level
source localization is supported by the observed failure mode and by the fact
that disabling HIP runtime tracing eliminates all profiler-level blockers.

## Remaining Gap

The 16 partial targets are no longer blocked by rocprofv3 collection itself.
Their remaining gaps are workload-level evaluation outcomes:

- `RUNTIME_ERROR` in generated inputs or reference/user execution.
- `INVALID_REFERENCE` for individual workloads.

The five previously excluded MoE targets remain out of scope for this quick
because dynamic timing-input preflight estimates exceed this host's realistic
memory limit.

## Verification

```text
uv run python -m py_compile scripts/run_rdna4_profiler_timing_batch.py
uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py
uv run pytest tests/sol_execbench/core/data/test_workload.py
```

Results:

- `tests/sol_execbench/test_rdna4_profiler_timing_batch.py`: `39 passed`
- `tests/sol_execbench/core/data/test_workload.py`: `8 passed`
