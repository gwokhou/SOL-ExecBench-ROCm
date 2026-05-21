---
phase: 03-rocm-evaluation-timing-and-hardware-introspection
plan: 01
subsystem: eval-driver
tags:
  - rocm
  - eval-driver
key-files:
  - src/sol_execbench/driver/templates/eval_driver.py
  - tests/sol_execbench/driver/test_eval_driver.py
metrics:
  tests: "15 passed"
---

# Plan 03-01 Summary

## Changes

- Updated eval-driver native shared-object routing to use Phase 2 ROCm language enums: `HIP_CPP`, `HIPBLAS`, `MIOPEN`, `CK`, and `ROCWMMA`.
- Updated GPU-server compile-blocking guidance from CUDA/C++ and `cuda_cpp` to HIP/native `hip_cpp`.
- Updated eval-driver schema regression coverage to use `hip_cpp` and `.hip` sources while preserving PyTorch's HIP-backed `at::cuda::getCurrentCUDAStream()` compatibility text.

## Verification

- `uv run --no-sync pytest tests/sol_execbench/driver/test_eval_driver.py` -> 15 passed.

## Deviations

None.

## Self-Check: PASSED

The eval driver now matches the ROCm schema names introduced in Phase 2 and keeps strict trace JSONL behavior covered by existing tests.
