# ROCm Library Category Readiness

This page defines the support status for ROCm library-oriented solution
categories. The schema recognizes these categories so future ROCm-native
solutions can state intent clearly, but recognition is not the same as runnable
public example coverage.

## Status Summary

| Category | Schema value | Current status | Public example status | Notes |
| --- | --- | --- | --- | --- |
| HIP/C++ | `hip_cpp` | Supported | Runnable HIP examples exist under `examples/hip_cpp/` | Primary native extension path. |
| hipBLAS / hipBLASLt | `hipblas` | Supported | Runnable SGEMM example exists under `examples/hipblas/gemm/` | BLAS/GEMM replacement path using the native ROCm build flow and `-lhipblas`. |
| MIOpen | `miopen` | Candidate | Former cuDNN examples are PyTorch compatibility examples | Use only after an operation-specific MIOpen solution is implemented and tested. |
| Composable Kernel | `ck` | Candidate | Former CUTLASS examples are PyTorch compatibility examples | Candidate replacement for selected CUTLASS/CuTe-style kernels. |
| rocWMMA | `rocwmma` | Candidate | No runnable public `rocwmma` example yet | Candidate for WMMA-style matrix kernels. |

## Support Levels

**Supported** means all of the following are true:

- the schema accepts the category,
- the build path is exercised by tests or examples,
- at least one public example uses the category directly,
- documentation describes dependencies and expected limitations,
- tests protect the example path and metadata.

**Candidate** means the schema has a ROCm-facing value and the category is a
known replacement direction, but public runnable examples and tests are not yet
sufficient to advertise full support.

**Compatibility example** means the example preserves the original problem
semantics using another supported runtime, usually PyTorch ROCm, while keeping
the old directory name for discoverability.

## Current Compatibility Examples

| Original NVIDIA category | Public path | Current implementation | Replacement direction |
| --- | --- | --- | --- |
| CUTLASS | `examples/cutlass/gemm/` | PyTorch ROCm compatibility example | `ck`, `rocwmma`, `hipblas`, or HIP/Triton |
| cuDNN | `examples/cudnn/softmax/` | PyTorch ROCm compatibility example | `miopen`, HIP, or Triton |
| CuTe DSL | `examples/cute_dsl/jamba_attn_proj/` | PyTorch ROCm compatibility example | `ck`, `rocwmma`, HIP, or Triton |
| cuTile | `examples/cutile/jamba_attn_proj/` | PyTorch ROCm compatibility example | HIP or Triton |

## Runnable Library Examples

| ROCm library | Public path | Evidence |
| --- | --- | --- |
| hipBLAS | `examples/hipblas/gemm/` | `solution_hipblas.json` uses schema language `hipblas`, includes `hipblas/hipblas.h`, calls `hipblasSgemm`, and links with `-lhipblas`. |

## User Guidance

- Use `hip_cpp`, `hipblas`, `pytorch`, or `triton` for runnable examples today.
- Use `miopen`, `ck`, or `rocwmma` only when the solution source and local ROCm
  environment actually provide the required library integration.
- Do not treat candidate categories as evidence of performance portability from
  NVIDIA libraries.
- CDNA 3 metadata in examples remains schema/build intent, not real hardware
  validation evidence.
