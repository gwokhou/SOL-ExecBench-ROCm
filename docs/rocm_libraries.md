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
| MIOpen | `miopen` | Supported | Runnable softmax example exists under `examples/miopen/softmax/` | cuDNN-style softmax replacement path using the native ROCm build flow and `-lMIOpen`. |
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
| cuDNN | `examples/cudnn/softmax/` | PyTorch ROCm compatibility example | `examples/miopen/softmax/`, HIP, or Triton |
| CuTe DSL | `examples/cute_dsl/jamba_attn_proj/` | PyTorch ROCm compatibility example | `ck`, `rocwmma`, HIP, or Triton |
| cuTile | `examples/cutile/jamba_attn_proj/` | PyTorch ROCm compatibility example | HIP or Triton |

## Runnable Library Examples

| ROCm library | Public path | Evidence |
| --- | --- | --- |
| hipBLAS | `examples/hipblas/gemm/` | `solution_hipblas.json` uses schema language `hipblas`, includes `hipblas/hipblas.h`, calls `hipblasSgemm`, and links with `-lhipblas`. |
| MIOpen | `examples/miopen/softmax/` | `solution_miopen.json` uses schema language `miopen`, includes `miopen/miopen.h`, calls `miopenSoftmaxForward_V2`, and links with `-lMIOpen`. |

The MIOpen softmax example is intentionally operation-specific: it validates
float32 tensors shaped `[batch_size, 4096]`, maps them to MIOpen tensor
descriptors as `N=batch_size, C=4096, H=1, W=1`, and uses
`MIOPEN_SOFTMAX_MODE_INSTANCE` so the measured implementation computes softmax
across the hidden dimension for each batch item.

## Dependency Diagnostics

The Docker dependency suite checks the development files needed by runnable
library examples. The expected ROCm package or file groups are:

| Category | Required headers | Required libraries | Typical ROCm packages |
| --- | --- | --- | --- |
| hipBLAS | `hipblas/hipblas.h` | `libhipblas.so` | `hipblas`, `hipblas-dev` |
| MIOpen | `miopen/miopen.h` | `libMIOpen.so` | `miopen-hip`, `miopen-hip-dev` |
| Composable Kernel | `ck/ck.hpp` | Header-only for the planned example path | `composablekernel-dev` or ROCm `rocm-libraries` headers |
| rocWMMA | `rocwmma/rocwmma.hpp` | Header-only for the planned example path | `rocwmma-dev`, `rocwmma` |

Missing headers and libraries are reported through
`sol_execbench.core.diagnostics.rocm_library_diagnostics()` so failures name the
specific dependency group instead of surfacing as opaque compile errors.

## User Guidance

- Use `hip_cpp`, `hipblas`, `miopen`, `pytorch`, or `triton` for runnable examples today.
- Use `ck` or `rocwmma` only when the solution source and local ROCm
  environment actually provide the required library integration.
- Do not treat candidate categories as evidence of performance portability from
  NVIDIA libraries.
- v1.8 library validation is scoped to RDNA 4 only. CDNA 3 and CDNA 4 metadata
  remains schema/build intent until real hardware evidence is recorded in a
  later milestone.
