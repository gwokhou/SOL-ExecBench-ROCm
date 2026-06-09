# Project Research - Features for RDNA4 Readiness Blocker Closure

## Scope

Milestone v1.34 targets the current 114 RDNA4 `readiness_blocked` problems:

- 55 `missing_evidence` problems caused by custom inputs.
- 33 `cuda_kernel_dependency` Quant problems caused by NVIDIA/CUDA runtime
  hints.
- 26 `flashinfer_runtime_assumption` FlashInfer-Bench problems.

The milestone should reduce readiness blockers by making more problems safe to
attempt on ROCm or by moving them into more precise residual blocker classes.

## Table Stakes

### Custom Input Readiness

- Evaluator can execute benchmark-defined custom input entrypoints safely.
- Generated inputs are deterministic per workload.
- Generated input keys, shapes, dtypes, scalar values, and devices are checked
  before execution.
- Failures are classified as input-generation errors or OOMs, not hidden as
  generic readiness blockers.
- Coverage reports show how many custom-input problems moved out of
  `readiness_blocked`.

### Quant Readiness

- Static CUDA/NVIDIA hint classification separates real runtime dependencies
  from false positives in names, comments, or compatibility labels.
- Quant semantic references that run under PyTorch ROCm can be marked ready or
  hardware-evidence-needed as appropriate.
- Low-precision validation boundaries are explicit: readiness can improve
  without claiming CDNA4 or unsupported low-precision hardware authority.
- Any true CUDA-only Quant path remains blocked with a precise next action.

### FlashInfer Readiness

- FlashInfer-Bench is split by semantic dependency instead of category-only
  blocking.
- Simple PyTorch-compatible workloads can move to ready execution.
- Paged, ragged, MLA, MoE, and FlashInfer-specific runtime workloads remain
  separately classified until a compatibility path exists.
- Coverage reports state which FlashInfer problems were reclassified and why.

### Coverage and Claim Closure

- RDNA4 coverage is recomputed after readiness changes.
- The blocker ledger records before/after classification deltas.
- Reports distinguish resolved readiness blockers from new runtime, OOM,
  correctness, profiler, or hardware-evidence blockers.
- Public docs and guardrails prevent paper-parity, leaderboard, CDNA3, or CDNA4
  claim upgrades.

## Differentiators

- A before/after readiness transition ledger for all 114 original blockers.
- Workload-level custom-input failure classes that preserve denominator
  accounting.
- Quant false-positive hint tests that prevent future overblocking.
- FlashInfer semantic taxonomy that can guide later high-performance ROCm work.

## Anti-Features

- Treating readiness reduction as validation success.
- Replacing custom inputs with random tensors.
- Making FlashInfer compatibility depend on unverified CUDA semantics.
- Reporting Quant low-precision readiness as CDNA4 hardware validation.
- Collapsing new runtime failures into generic readiness blockers.

