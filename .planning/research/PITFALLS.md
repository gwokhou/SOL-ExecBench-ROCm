# Project Research - Pitfalls for RDNA4 Readiness Blocker Closure

## Custom Input Pitfalls

### Random Substitution

Replacing custom inputs with random tensors would produce meaningless
correctness results. Many blocked problems require semantically coupled values
such as dropout masks, routing indices, position IDs, KV-cache updates, or
variable-length metadata.

Prevention: require the benchmark-defined custom entrypoint and validate its
outputs before execution.

### Nondeterminism

Custom input functions can call random tensor generation. If seeds are not
controlled per workload, failures and coverage deltas will be hard to reproduce.

Prevention: set deterministic seeds around entrypoint execution and record seed
policy in evidence.

### Hidden OOM Conversion

Custom input generation can OOM before reference execution. If this is reported
as generic readiness failure, the milestone will not improve diagnosability.

Prevention: classify `gen_inputs_oom_blocked` and separate it from reference
and user-solution OOM.

## Quant Pitfalls

### False Positive CUDA Hints

Names like `CuBLASRefBlockwiseGemm` or variables like `scale_w_cublas` may
describe compatibility layout rather than actual CUDA runtime use.

Prevention: make static hint classification context-aware and back it with
fixtures.

### Low-Precision Claim Leakage

Moving Quant references to ready can be mistaken for CDNA4 or full
low-precision hardware validation.

Prevention: keep readiness, semantic compatibility, hardware evidence, and
performance authority as separate fields and docs.

### Real Runtime Dependencies

Some Quant paths may still contain true CUDA-only code or NVIDIA DSL
dependencies.

Prevention: keep precise `rocm_port_needed` blockers with source evidence and
next actions.

## FlashInfer Pitfalls

### Category-Only Overblocking

Blocking every FlashInfer-Bench problem hides simple PyTorch-compatible cases.

Prevention: classify by semantic dependency, not by category alone.

### Unsafe Semantic Approximation

Paged/ragged decode and prefill workloads depend on layout and runtime
semantics that a simple PyTorch rewrite may not preserve.

Prevention: only release those blockers after compatibility helpers encode page
tables, offsets, masks, KV layout, and dtype assumptions.

### Performance Scope Creep

Readiness closure can drift into high-performance FlashInfer kernel tuning.

Prevention: limit v1.34 to correctness/readiness and claim-safe execution
closure; defer performance tuning.

## Reporting Pitfalls

### Readiness Reduction as Validation Claim

Moving a problem out of `readiness_blocked` means it can be attempted, not that
it passed.

Prevention: coverage reports must show transitions into pass, fallback,
runtime/OOM/correctness/profiler blockers, or residual deferrals.

### Denominator Drift

Resolving readiness blockers must not change the 235-problem denominator.

Prevention: tests should assert denominator stability and before/after blocker
accounting.

