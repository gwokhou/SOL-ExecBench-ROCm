# Phase 137 Summary

## Status

Completed on 2026-06-07.

## Delivered

- Created the RDNA4 long-running validation runbook with preflight, polling,
  checkpoint, resume, classification, and claim-boundary rules.
- Recorded local host preflight evidence: host tools see RDNA4 `gfx1200`
  (`rocminfo`) and Navi 44 / Radeon RX 9060 XT (`lspci`).
- Ran RDNA4 marker pytest. The selected tests skipped because the `uv`/pytest
  environment could not see `/dev/kfd` or `/dev/dri`, even though the outer
  shell can see device nodes. This is classified as an execution-environment
  passthrough boundary, not RDNA4 pass evidence and not a GPU correctness
  failure.
- Ran focused category/example evidence:
  - category guardrails: 38 passed;
  - category examples: 9 passed, 9 skipped for the same device-node boundary.
- Restored README guardrail wording for CDNA 3 marker context and deferred
  CDNA 4 validation.
- Added a CPU-safe guardrail ensuring Phase 137 category evidence cannot imply
  full dataset validation, benchmark-grade timing, public claim upgrade,
  CDNA3/MI300X validation, or CDNA4 validation.

## Boundaries

- Phase 137 does not complete the Phase 138 full dataset run.
- Phase 137 does not establish Phase 139 timing authority.
- Skipped GPU-marker tests are not validation passes.
- The unrelated NVIDIA GPU visible in `lspci` is not evidence for this
  ROCm/RDNA4 milestone.
