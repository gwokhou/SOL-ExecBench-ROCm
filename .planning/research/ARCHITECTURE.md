# Project Research - Architecture for RDNA4 Readiness Blocker Closure

## Existing Architecture Context

The current pipeline is:

1. Local migrated dataset under `data/SOL-ExecBench/benchmark`.
2. Dataset inventory parses definitions, references, workloads, and solution
   hints.
3. ROCm readiness classifies each workload/problem before execution.
4. Dataset execution/profiler scripts attempt ready workloads and write traces,
   timing sidecars, closure reports, and coverage summaries.
5. Coverage reports combine readiness and timing evidence into claim-safe
   totals over the 235-problem denominator.

Milestone v1.34 should add execution readiness capabilities without weakening
that architecture.

## Proposed Data Flow

### Custom Input Flow

1. Inventory records `custom_inputs_entrypoint` and custom input fields.
2. Readiness treats supported custom input entrypoints as conditionally ready
   instead of blocked.
3. Execution resolves and imports the reference source in isolation.
4. Execution calls the custom input entrypoint with workload axes/scalars and
   the selected ROCm device.
5. Generated inputs are validated against definition/workload metadata.
6. Reference and candidate execution use the generated inputs.
7. Failures are classified as:
   - input generation failure,
   - input generation OOM,
   - reference OOM,
   - candidate/runtime failure,
   - correctness failure,
   - profiler failure.

### Quant Flow

1. Inventory records runtime hints with source/context when possible.
2. Readiness distinguishes real CUDA imports/calls from lexical false positives.
3. PyTorch ROCm semantic references move to ready or hardware-evidence-needed.
4. True CUDA-only paths remain `rocm_port_needed`.
5. Coverage reports carry the refined blocker class.

### FlashInfer Flow

1. Readiness no longer treats all `FlashInfer-Bench` problems as one class.
2. Static classifier assigns semantic buckets:
   - PyTorch-compatible normalization/GEMM/simple fused ops,
   - paged decode/prefill,
   - ragged prefill,
   - MLA,
   - MoE/FP8 block-scale,
   - unknown FlashInfer runtime dependency.
3. PyTorch-compatible buckets can be attempted.
4. True runtime buckets remain blocked until compatibility helpers exist.

## Build Order

1. Add classifier/reporting improvements that expose current readiness subtypes
   and transition accounting.
2. Implement custom input generation support because it affects the largest
   blocker class.
3. Triage Quant hints and low-precision boundaries.
4. Split FlashInfer readiness and implement the safe PyTorch-compatible subset.
5. Recompute coverage and update claim documentation.

## Modified Components

| Component | Change |
| --- | --- |
| Dataset readiness | Add supported custom-input, Quant hint, and FlashInfer semantic classifications |
| Eval/input assembly | Add deterministic custom input entrypoint execution |
| Execution closure | Record new failure classes after readiness blockers are attempted |
| Profiler coverage | Add before/after readiness transition ledger |
| Docs/tests | Preserve claim boundaries and regression coverage |

