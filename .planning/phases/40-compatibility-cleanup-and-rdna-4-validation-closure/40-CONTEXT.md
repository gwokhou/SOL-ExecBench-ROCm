---
phase: 40
name: Compatibility Cleanup and RDNA 4 Validation Closure
status: complete
created: 2026-05-22
updated: 2026-05-22
---

# Phase 40 Context

## Goal

Remove remaining ambiguity around former NVIDIA compatibility examples, native
ROCm library support claims, and RDNA 4-only validation scope.

## Relevant Inputs

- v1.8 requirement groups: `COMPAT-01` through `COMPAT-04` and `RDNA4-01`
  through `RDNA4-03`
- User scope: v1.8 validates on RDNA 4 only; CDNA 3 and CDNA 4 validation is deferred
- Supported library examples delivered in Phases 36-39:
  - `examples/hipblas/gemm/`
  - `examples/miopen/softmax/`
  - `examples/ck/gemm/`
  - `examples/rocwmma/gemm/`

## Validation Boundary

The focused unit/docs/staging suite passes locally. Native library E2E examples
are registered and guarded for RDNA 4, complete ROCm native extension headers,
and the corresponding library headers. On this local machine, MIOpen, CK, and
rocWMMA E2E tests are skipped because the full development header set is
incomplete.
