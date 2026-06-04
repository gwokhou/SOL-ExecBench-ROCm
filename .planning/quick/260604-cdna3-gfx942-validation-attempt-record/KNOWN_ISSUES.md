---
status: active
updated_at: "2026-06-05"
slug: cdna3-gfx942-known-issues
---

# CDNA3 gfx942 Known Issues

This file records the known issues from the CDNA3/gfx942 dataset validation and
derived-scoring runs so the investigation state is not lost across sessions.

## Context

Latest code inspected after pulling user performance work:

- Branch: `main`
- Head: `aa78dcb #0 - Optimize heavy benchmark scripts`
- Prior local fixes already merged and pushed:
  - `7e7d564 Add phased dataset validation workflow`
  - `efeb10e Preserve traces from nonzero dataset CLI exits`

Primary run shape:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --category L1 L2 Quant \
  -o out/cdna3-sol-validation \
  --gpu-architecture gfx942 \
  --timing-evidence-dir out/cdna3-sol-timing \
  --amd-score-report out/cdna3-sol-amd-score.json \
  --amd-sol-bound-dir out/cdna3-sol-amd-sol-bounds \
  --solar-derivation out/cdna3-sol-solar-derivation \
  --dataset-manifest out/sol-migration-manifest.json
```

User-provided full artifact archive:

- `~/Downloads/out.tar`

## Fixed During This Debug Cycle

### SOLAR parser rejected nested JSON-compatible values

Root cause:

- `src/sol_execbench/core/scoring/solar_derivation.py` accepted only scalar object
  map values in formula inputs.
- Some evidence contains recursive JSON values, for example
  `formula_inputs.axis: [1, 2]`, which caused parse failure.

Fix:

- `_parse_object_map` now accepts recursive JSON-compatible values via
  `_ensure_json_value`.
- Regression coverage added with
  `test_formula_inputs_round_trip_multi_axis_json_values`.

### Dataset runner lost usable traces when CLI exited nonzero

Root cause:

- Static-review failures such as `REWARD_HACK` can emit valid JSONL trace records
  to stdout and then exit with code `1`.
- The dataset runner treated the nonzero exit as no usable trace output.

Fix:

- `src/sol_execbench/core/dataset/runner.py` now preserves parsed stdout traces
  from nonzero CLI exits.
- This allows recoverable `REWARD_HACK` evidence to be included in later derived
  phases instead of losing the whole problem's progress.

## Current Validation Result Summary

Observed from the full validation artifact:

- Problems total: `209`
- Problem status: `170 OK`, `39 FAIL`
- Workloads total: `3106`
- Passed workloads: `2694`
- Failed workloads: `424`
- Trace files: `197`
- CLI logs: `39`

The `39 FAIL` problems are not one single failure mode. They currently split into
timeouts, static-review blocks, and ROCm-incompatible Quant references.

## Current Timing Phase Result Summary

Observed from `out/cdna3-sol-timing.log`:

- Discovered problems: `209`
- Timing phase processed problems with existing traces: `197`
- Timing phase summary: `170 OK`, `27 FAIL`
- Skipped problems: `12`
- Skip reason: `--phase timing requires existing traces`

Important interpretation:

- The timing phase reused existing traces and did not regenerate validation
  traces.
- The `12` skipped problems are the same no-trace timeout cases listed below.
- Therefore the timing summary denominator is `197`, not `209`.
- The difference between validation `39 FAIL` and timing `27 FAIL` is explained
  by those `12` skipped no-trace problems.

Timing-phase failed problems:

```text
L1/013_fused_residual_rms_norm_backward
L1/018_fused_rope_with_qk_norm_and_kv_cache_update
L1/051_attention_qkv_with_qk_norm_single_kernel_backward
L1/062_kv_cache_update_with_rope_backward
L1/071_kv_cache_update_with_rope
L2/033_multi_scale_feature_pyramid
L2/040_altup_predict_correction_cycle_backward
L2/044_mamba_discretization_and_segsum
L2/056_language_model_decoder_prenorm_attention_ffn_residual_backward
L2/080_moe_complete_layer_with_shared_expert_backward
Quant/008_nvfp4_multimodal_embedding_projection
Quant/009_nvfp4_vision_temporal_patch_merge_with_projection
Quant/011_fp8_moe_gate_routing
Quant/018_nvfp4_attention_output_projection_with_residual
Quant/019_nvfp4_grouped_query_attention
Quant/020_nvfp4_linear_layer
Quant/023_fp8_mamba2_ssm_discretization
Quant/024_nvfp4_attention_output_projection
Quant/025_nvfp4_attention_qkv_projection
Quant/026_nvfp4_mamba2_out_projection
Quant/027_nvfp4_moe_router_projection
Quant/028_nvfp4_attention_qkv_projection_with_rope
Quant/029_nvfp4_fused_mlp_silu_gating
Quant/030_nvfp4_fused_decoder_layer_pre_norm_attention
Quant/031_nvfp4_grouped_query_attention_with_kv_repeat
Quant/032_nvfp4_moe_expert_linear_with_gating
Quant/033_nvfp4_moe_routing_with_topk_selection
```

Additional timing-specific note:

- `L2/033_multi_scale_feature_pyramid` appears as a partial failure in timing
  summary: `7` pass, `9` fail. This should be inspected separately because it is
  not one of the all-workload static-review or Quant scaled-GEMM failures.

Follow-up classification from trace inspection:

- `L2/033_multi_scale_feature_pyramid` is not a timing orchestration failure.
- Its `traces.json` already contains `Counter({'INCORRECT_NUMERICAL': 9,
  'PASSED': 7})`.
- The timing phase merely reused those existing traces and reflected the same
  correctness status.
- Passing workloads report exact equality: `max_absolute_error = 0.0`.
- Failing workloads report very large absolute errors around `442368` to
  `540672`, with no NaN/Inf.
- The benchmark reference is a pure PyTorch float32 conv chain:
  `conv2d -> conv2d stride 2 -> conv2d -> conv2d stride 2 -> conv2d ->
  conv2d -> conv_transpose2d -> cat -> conv2d -> conv_transpose2d -> cat ->
  conv2d -> conv2d`.
- Because the run uses `Definition.reference` as the solution, this points to a
  reference-vs-reference reproducibility issue for selected ROCm/MIOpen shapes,
  or an evaluation input/output reuse issue, rather than a candidate kernel bug.

Follow-up direct-reference reproducibility check:

- A remote script imported
  `data/SOL-ExecBench/benchmark/L2/033_multi_scale_feature_pyramid/reference.py`
  directly and compared two consecutive `run()` calls on the same generated
  inputs for all `16` workloads.
- Result: `Counter({'PASS': 16})`.
- Largest direct-reference absolute differences were small, about `4.7e-05` to
  `5.9e-05`, with matched ratio `1.0`; several shapes were exactly equal.
- This weakens the MIOpen nondeterminism hypothesis.
- Current strongest hypothesis: the failure is in the eval-driver packaging or
  comparison path, for example `_reference.py` versus staged `reference.py`,
  solution loading, generated input ordering, or input/output reuse around
  `call_and_collect_outputs`.

Root cause identified:

- `L2/033` has `custom_inputs_entrypoint: "get_inputs"` in `definition.json`.
- Its workloads mark inputs as `"type": "random"` rather than `"type": "custom"`.
- Before the fix, `gen_inputs()` called the custom factory but only used its
  returned values for workload inputs explicitly marked `CustomInput`.
- Therefore the benchmark ignored `reference.py::get_inputs()` for `L2/033` and
  generated generic random conv weights instead of the intended Kaiming-scaled
  weights.
- The generic random weights make the multi-layer conv pyramid numerically
  ill-conditioned enough that reference-vs-reference runs through separate
  eval-driver modules can exceed the strict tolerance.

Fix applied locally:

- `src/sol_execbench/core/bench/io.py`: when a custom input factory returns a
  value for an input name, `gen_inputs()` now uses that value even if the
  workload input is marked `random`.
- Explicit safetensors and scalar workload inputs still take precedence.
- Added focused regression test:
  `TestGenInputs::test_custom_factory_values_override_random_workload_inputs`.

Timing FAIL classes are now:

- Static review `REWARD_HACK`: `12` problems.
- Quant CUDA-only scaled GEMM `INVALID_REFERENCE`: `14` problems.
- `L2/033_multi_scale_feature_pyramid` partial numerical failure: `1` problem.

## Issue 1: Timeout Problems Without Traces

These problems produced no `traces.json` because the internal evaluation driver
hit the 300 second timeout:

```text
L1/076_batched_expert_forward
L1/094_time_decay_exponential_stabilization
L2/004_fused_residual_rms_mlp
L2/005_swiglu_mlp_backward
L2/023_video_latent_vae_encoder_downsampling
L2/024_moe_expert_parallel_execution
L2/025_moe_expert_parallel_execution_backward
L2/026_moe_expert_parallel_execution_with_weighted_aggregation
L2/047_moe_training_token_repeat_and_expert_computation
L2/055_audio_encoder_conv_positional_layer_stack
L2/077_sam_hq_vision_encoder_window_partition_attention
L2/078_fused_final_layer_upsample_with_adaptive_norm
```

Current interpretation:

- These are benchmark/runtime execution blockers, not derived parser failures.
- Recovery is possible only by rerunning the trace phase for these specific
  problems after changing timeout, workload selection, or implementation.
- Because there is no trace file, downstream derived and timing phases cannot
  reconstruct missing workload evidence for them.

## Issue 2: Static Source Review Blocks

Several reference implementations are blocked as `REWARD_HACK` by static source
review. The two dominant rules seen so far are:

- `precision_downgrade`
- `semantic_output_cache`

Examples:

```text
L1/013_fused_residual_rms_norm_backward
L1/018_fused_rope_with_qk_norm_and_kv_cache_update
L1/051_attention_qkv_with_qk_norm_single_kernel_backward
L1/062_kv_cache_update_with_rope_backward
L1/071_kv_cache_update_with_rope
L2/040_altup_predict_correction_cycle_backward
L2/044_mamba_discretization_and_segsum
L2/056_language_model_decoder_prenorm_attention_ffn_residual_backward
L2/080_moe_complete_layer_with_shared_expert_backward
Quant/011_fp8_moe_gate_routing
Quant/023_fp8_mamba2_ssm_discretization
Quant/033_nvfp4_moe_routing_with_topk_selection
```

Representative evidence:

- `precision_downgrade` on expressions such as `grad_x.to`,
  `grad_q_normed.to`, `grad_k_normed.to`, and `torch.bfloat16`.
- `semantic_output_cache` on cache-related setup such as `cache_len`,
  `key_cache`, `value_cache`, and `cache_position`.

Current interpretation:

- These are policy/static-review outcomes from benchmark reference code under
  ROCm, not failures introduced by the phased dataset workflow.
- Since stdout traces are now preserved for nonzero exits, these can still
  contribute explicit blocked evidence to derived outputs.
- Each rule hit needs a separate decision: either fix the reference to comply,
  tune the static-review heuristic if it is a false positive, or record the case
  as intentionally ineligible.

## Issue 3: Quant NVFP4/FP8 CUDA-Only References

Several Quant problems fail as `INVALID_REFERENCE` because the reference path uses
CUDA-only scaled GEMM support:

```text
scaled_gemm with torch.float8_e4m3fn scales of 1x16 blocks is only supported for CUDA 12.8 and above
```

Affected examples include:

```text
Quant/008
Quant/009
Quant/018
Quant/019
Quant/020
Quant/024
Quant/025
Quant/026
Quant/027
Quant/028
Quant/029
Quant/030
Quant/031
Quant/032
```

Current interpretation:

- This is a ROCm portability gap in reference execution, not a dataset
  orchestration problem.
- These cases need ROCm-compatible reference behavior, a skip/ineligible policy,
  or a documented limitation.

## Issue 4: Derived Score Has Many Unsupported/Ineligible Results

Derived/scoring output exists and is structurally complete:

- SOLAR sidecars: `3106`
- AMD SOL bound sidecars: `3106`
- Score report exists: yes
- `scores`: `3106`
- `scored_count`: `196`
- `unscored_count`: `2910`
- `unscored[]`: `0`

Important nuance:

- The report does not use a separate `unscored[]` list.
- Unscored entries are represented inside `scores[]` with
  `supported: false`.

Observed counters:

```text
supported: Counter({'False': 2910, 'True': 196})
claim_level: Counter({'amd-native-derived': 3106})
family: Counter({None: 3106})
status: Counter({None: 3106})
```

Top warnings/reasons:

```text
3106 model_validation:gfx1200:provisional
2910 aggregate_unscored:unsupported operation evidence present
2910 AMD-native score was not computed because AMD SOL bound evidence is marked unscored.
2715 aggregate_degraded:incomplete semantic evidence
2694 AMD-native score used PyTorch reference latency as a provisional baseline; provide a scoring baseline artifact for release-defined scoring.
2567 aggregate_unscored:unsupported semantic evidence
2567 unsupported_operator:unsupported
2299 inexact_operator:elementwise
2174 inexact_operator:data_movement
1717 unsupported_operator:op_2:unsupported
1673 unsupported_operator:op_1:unsupported
1610 inexact_operator:dtype_conversion
1438 graph_warning:dynamic_trace_failed
1267 graph_warning:unsupported_operator:getitem
1101 inexact_operator:reduction
1090 unsupported_operator:getitem
832 graph_warning:unsupported_operator:torch.cat
```

Current interpretation:

- The derived phase now completes, but scoring coverage is low:
  `196 / 3106` scored.
- Main blockers are incomplete semantic evidence and unsupported/inexact operator
  extraction.
- This is a model/evidence coverage issue, not a parser or phase-runner crash.

## Issue 5: Hardware Model Falls Back To gfx1200

User observed all trace environments report generic hardware:

```text
Counter({'AMD Radeon Graphics': 3106})
{'AMD Radeon Graphics': 'out/cdna3-sol-validation/L1/001_attention_softmax_dropout_value_matmul_backward/traces.json'}
```

Current interpretation:

- Trace environment records only generic `"AMD Radeon Graphics"`.
- Packaged AMD hardware models currently include `gfx1200`, so derived AMD SOL
  evidence and score warnings use `model_validation:gfx1200:provisional`.
- `--gpu-architecture gfx942` is not enough to produce true `gfx942` AMD SOL
  scoring unless a `gfx942` hardware model is added and selected in derived
  scoring.
- Until then, CDNA3 score numbers should be treated as provisional and not
  release-grade.

## Timing Semantics

The phased workflow does not intentionally change benchmark timing semantics:

- `--phase all` preserves the original user-facing CLI behavior.
- `--phase traces` runs validation/evaluation and writes trace files.
- `--phase derived` is CPU/IO-only and may use `--jobs`; it reads existing traces
  and does not run GPU benchmark timing.
- `--phase timing` remains GPU/profiler-bound and should stay serial for reliable
  timing semantics.

Any timing comparison should use the `timing` phase, not derived-phase wall-clock
speed.

## Recovery Notes

Recoverable artifacts:

- Existing `traces.json` files can be reused by `--phase derived` and
  `--phase timing`.
- Nonzero CLI exits that emitted valid JSONL traces are now preserved.
- Derived outputs can be regenerated without rerunning GPU validation.

Not recoverable without rerun:

- Problems that timed out before producing `traces.json`.
- Workloads whose reference failed before usable trace records were emitted.

Recommended recovery pattern:

1. Keep the existing `out/cdna3-sol-validation` tree.
2. Rerun `--phase traces` only for missing or intentionally targeted problem
   subsets when narrowing failures.
3. Rerun `--phase derived --jobs auto` after trace fixes.
4. Rerun `--phase timing` only after validation trace coverage is acceptable.

## Recommended Next Commands

Regenerate derived artifacts from existing traces:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --phase derived \
  --jobs auto \
  --category L1 L2 Quant \
  -o out/cdna3-sol-validation \
  --gpu-architecture gfx942 \
  --amd-score-report out/cdna3-sol-amd-score.json \
  --amd-sol-bound-dir out/cdna3-sol-amd-sol-bounds \
  --solar-derivation out/cdna3-sol-solar-derivation \
  --dataset-manifest out/sol-migration-manifest.json \
  | tee -a out/cdna3-sol-derived.log
```

Collect timing evidence from existing traces:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --phase timing \
  --category L1 L2 Quant \
  -o out/cdna3-sol-validation \
  --gpu-architecture gfx942 \
  --timing-evidence-dir out/cdna3-sol-timing \
  --dataset-manifest out/sol-migration-manifest.json \
  | tee -a out/cdna3-sol-timing.log
```

## Open Decisions

- Add a real `gfx942` AMD hardware model, or explicitly document that current
  derived scores use provisional `gfx1200` modeling.
- Decide policy for ROCm-incompatible Quant scaled GEMM references.
- Decide whether `precision_downgrade` and `semantic_output_cache` hits are true
  benchmark violations or static-review false positives for migrated references.
- Improve semantic evidence extraction coverage for common unsupported operators
  such as `getitem`, `torch.cat`, dynamic trace failures, and inexact data
  movement/dtype conversion.
- Decide whether timeout-heavy problems should get per-problem timeout overrides,
  workload slicing, or deferred unsupported status for CDNA3 validation.
