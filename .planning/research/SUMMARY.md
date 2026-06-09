# Project Research Summary - RDNA4 Readiness Blocker Closure

## Stack Additions

- Add deterministic custom input entrypoint execution in the evaluator/input
  assembly path.
- Refine Quant static CUDA/NVIDIA hint classification so true runtime
  dependencies and lexical false positives are separated.
- Split FlashInfer-Bench readiness by semantic dependency instead of blocking
  the whole category.
- Extend RDNA4 coverage reports with before/after readiness transition
  accounting.

## Feature Table Stakes

- The 55 custom-input blockers need benchmark-defined input generation support,
  not random tensor substitution.
- The 33 Quant blockers need static hint triage plus explicit low-precision
  evidence boundaries.
- The 26 FlashInfer blockers need a semantic split: PyTorch-compatible simple
  cases can move forward, true FlashInfer runtime cases need dedicated
  compatibility work or residual blockers.
- Recomputed coverage must preserve the 235-problem denominator and show where
  each original readiness blocker moved.

## Watch Out For

- Do not report readiness reduction as validation success.
- Do not let Quant readiness imply CDNA4 or low-precision hardware authority.
- Do not let RDNA4 readiness evidence upgrade CDNA3 or CDNA4 claims.
- Do not hide new input-generation OOM, reference OOM, runtime, correctness, or
  profiler failures under generic readiness labels.

## Recommended Build Order

1. Add readiness transition accounting and custom-input support.
2. Recompute custom-input coverage and classify new blockers.
3. Refine Quant hint detection and low-precision readiness boundaries.
4. Split FlashInfer-Bench into PyTorch-compatible and true runtime-dependent
   subsets.
5. Regenerate coverage, blocker ledgers, docs, and claim guardrails.

