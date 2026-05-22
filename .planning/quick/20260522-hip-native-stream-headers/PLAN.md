---
quick_task: hip-native-stream-headers
status: complete
created: 2026-05-22
---

# HIP Native Stream Headers

## Goal

Replace CUDA compatibility headers and stream/guard APIs in public native
examples with HIP-specific PyTorch headers and APIs.

## Tasks

- [x] Replace `ATen/cuda/CUDAContext.h` usage in native examples.
- [x] Replace `at::cuda::getCurrentCUDAStream()` with
  `c10::hip::getCurrentHIPStream()`.
- [x] Replace `c10/cuda/CUDAGuard.h` usage in the Flux RoPE example with
  `c10/hip/HIPGuard.h`.
- [x] Sync embedded `solution_*.json` source content with checked-in source
  files.
- [x] Run focused example consistency and native library staging tests.
