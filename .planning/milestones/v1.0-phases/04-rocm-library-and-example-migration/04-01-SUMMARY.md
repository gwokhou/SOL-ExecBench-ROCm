---
phase: 04-rocm-library-and-example-migration
plan: 01
subsystem: hip-examples
tags:
  - rocm
  - examples
key-files:
  - examples/cuda_cpp/flux_rope/solution_cuda.json
  - examples/cuda_cpp/flux_rope/kernel.hip
  - examples/cuda_cpp/rmsnorm/solution_cuda.json
  - examples/cuda_cpp/rmsnorm/kernel.hip
  - tests/sol_execbench/samples/flux_rope/solution_cuda.json
  - tests/sol_execbench/samples/rmsnorm/solution_cuda.json
metrics:
  tests: "11 passed"
---

# Plan 04-01 Summary

## Changes

- Migrated CUDA C++ example metadata to `hip_cpp`, `gfx1200`, `.hip` source paths, and `hip_cflags`.
- Renamed public example kernel files from `.cu` to `.hip`.
- Replaced obvious NVIDIA-only headers and type names in public HIP example sources with HIP headers/types where simple.
- Updated PyTorch/Triton sample targets from `B200` to `gfx1200`.

## Verification

- `uv run --no-sync pytest tests/examples/test_examples.py -k consistency tests/sol_execbench/test_rocm_library_examples.py` -> 11 passed.

## Deviations

- Existing filenames such as `solution_cuda.json` are retained for compatibility with existing example tests, but the JSON contents now use ROCm schema values.

## Self-Check: PASSED

HIP/C++ example and sample metadata parses under the ROCm schema and references existing source files.
