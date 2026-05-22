---
phase: 39
name: rocWMMA Supported Replacement
status: complete
created: 2026-05-22
updated: 2026-05-22
---

# Phase 39 Context

## Goal

Promote rocWMMA from candidate status to a supported ROCm library replacement
path for RDNA 4 matrix-core GEMM-style workloads.

## Relevant Inputs

- v1.8 requirement group: `WMM-01` through `WMM-04`
- User scope: v1.8 validates on RDNA 4 only; CDNA 3 and CDNA 4 validation is deferred
- Existing compatibility paths: `examples/cutlass/gemm/` and
  `examples/cute_dsl/jamba_attn_proj/`
- rocWMMA dependency diagnostics from Phase 36
- rocWMMA API reference for `fragment`, `load_matrix_sync`, `mma_sync`, and
  `store_matrix_sync`

## Implementation Boundary

The measured implementation must call rocWMMA APIs directly. PyTorch is retained
only for the benchmark reference implementation and tensor/extension
integration.

The public example targets `LOCAL` and `gfx1200`; it does not claim CDNA 3 or
CDNA 4 validation.
