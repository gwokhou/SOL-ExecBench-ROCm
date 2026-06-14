---
quick_id: 260613-sp3
slug: readiness-blocked-quant-readiness-blocke
status: in_progress
created_at: 2026-06-13T12:39:42.941Z
---

# Quick Task: Close Quant And FlashInfer Readiness Blockers

## Goal

Close the current `readiness_blocked` gap for `Quant/*` and `FlashInfer-Bench/*`
in the RDNA4 validation denominator.

## Scope

- Analyze current `readiness_blocked` rows for `Quant` and `FlashInfer-Bench`
  from the latest merged artifact.
- For `nvidia_cuda_runtime_hint`, first determine whether a practical ROCm
  ecosystem counterpart exists.
- If a ROCm counterpart exists, migrate or classify toward the ROCm path.
- If no ROCm counterpart exists, preserve the blocker as an explicit annotated
  dependency instead of marking it execution-ready.
- For MXFP4/NVFP4 or CDNA4-only low-precision validation paths, migration and
  static classification may be completed, but local RDNA4 hardware validation
  remains deferred.

## Verification

- Unit tests for readiness and low-precision classification.
- Regenerate or inspect the latest merged readiness/coverage output to quantify
  remaining gaps by reason.
