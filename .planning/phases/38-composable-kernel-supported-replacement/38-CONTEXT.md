---
phase: 38
name: Composable Kernel Supported Replacement
status: complete
created: 2026-05-22
updated: 2026-05-22
---

# Phase 38 Context

## Goal

Promote Composable Kernel from candidate status to a supported ROCm library
replacement path for selected GEMM workloads, scoped to RDNA 4 validation.

## Relevant Inputs

- v1.8 requirement group: `CK-01` through `CK-04`
- User scope: v1.8 validates on RDNA 4 only; CDNA 3 and CDNA 4 validation is deferred
- Existing compatibility path: `examples/cutlass/gemm/`
- CK dependency diagnostics from Phase 36
- CK documentation around tensor descriptors and tile-oriented programming

## Implementation Boundary

The measured implementation is native HIP/C++ and does not use PyTorch for the
measured GEMM. PyTorch remains only the benchmark reference and tensor/extension
integration layer.

The public example targets `LOCAL` and `gfx1200`; it does not claim CDNA 3 or
CDNA 4 validation.
