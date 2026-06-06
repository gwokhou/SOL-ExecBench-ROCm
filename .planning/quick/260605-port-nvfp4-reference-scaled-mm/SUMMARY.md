# Port NVFP4 Reference Scaled MM Summary

## Status

Abandoned.

## Resolution

This task was superseded by the CDNA3 low-precision skip policy. The original
plan would have replaced CUDA-only `torch._scaled_mm` NVFP4/MXFP4 reference
paths with a portable dequantized matmul fallback. That approach is not being
implemented because it risks changing benchmark semantics for formats whose
hardware behavior is the benchmark subject.

Current policy:

- CDNA3 runs skip NVFP4/MXFP4 Quant hardware-validation problems with the
  `cdna3_low_precision_hardware_unsupported` reason.
- NVFP4/MXFP4 ROCm adaptation and benchmark hardware validation are deferred
  until CDNA4-class hardware is available.
- CPU-safe helpers remain semantic compatibility utilities only; they are not
  benchmark validation or performance authority.
