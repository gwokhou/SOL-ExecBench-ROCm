---
phase: 37
name: MIOpen Supported Replacement
status: complete
created: 2026-05-22
updated: 2026-05-22
---

# Phase 37 Context

## Goal

Promote MIOpen from candidate status to a supported ROCm library replacement
path for the former cuDNN softmax-style example, scoped to RDNA 4 validation.

## Relevant Inputs

- v1.8 requirement group: `MIOPEN-01` through `MIOPEN-04`
- User scope: v1.8 validates on RDNA 4 only; CDNA 3 and CDNA 4 validation is deferred
- Existing compatibility path: `examples/cudnn/softmax/`
- Existing native library precedent: `examples/hipblas/gemm/`
- MIOpen API reference: `miopenSoftmaxForward_V2`,
  `MIOPEN_SOFTMAX_ACCURATE`, and `MIOPEN_SOFTMAX_MODE_INSTANCE`

## Implementation Boundary

The measured implementation must call MIOpen directly. PyTorch is retained only
for the benchmark reference implementation and tensor/extension integration.

The public example targets `LOCAL` and `gfx1200`; it does not claim CDNA 3 or
CDNA 4 validation.
