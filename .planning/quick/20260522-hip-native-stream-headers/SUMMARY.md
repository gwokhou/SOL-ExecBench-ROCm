---
quick_task: hip-native-stream-headers
status: complete
completed: 2026-05-22
---

# Summary

Updated public native examples to use HIP-specific PyTorch headers and stream
APIs instead of CUDA compatibility headers.

## Changed

- `examples/hip_cpp/rmsnorm/main.cpp`
- `examples/hip_cpp/flux_rope/kernel.hip`
- `examples/hipblas/gemm/main.cpp`
- `examples/miopen/softmax/main.cpp`
- `examples/ck/gemm/kernel.hip`
- `examples/rocwmma/gemm/kernel.hip`
- Embedded `sources[].content` fields in the corresponding solution JSON files
- `src/sol_execbench/driver/templates/build_ext.py` now constrains
  `PYTORCH_ROCM_ARCH` from solution `gfx*` targets when the environment does
  not already set it, preventing unsupported default arch fan-out for rocWMMA.
- `tests/conftest.py` now treats ROCm dev availability as HIP runtime headers,
  not CUDA runtime headers.

## Verification

```bash
uv run pytest tests/examples/test_examples.py -k "hip or hipblas or miopen or ck or rocwmma"
```

Result: 16 passed.

```bash
uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py -k "hipblas or miopen or ck or rocwmma or source or stage or runnable"
```

Result: 15 passed.

```bash
uv run pytest tests/examples/test_examples.py -k "gemm_rocwmma or gemm_ck or consistency"
```

Result: 17 passed.

```bash
uv run --with ruff ruff check src/sol_execbench/driver/templates/build_ext.py tests/conftest.py
```

Result: all checks passed.
