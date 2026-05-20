# Stack Research

**Domain:** ROCm port of GPU kernel benchmark framework
**Researched:** 2026-05-21
**Confidence:** HIGH for official ROCm stack components, MEDIUM for one-to-one replacement completeness

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| ROCm | >= 7.0 | AMD GPU software baseline | User requirement; official matrix covers 7.0.x component versions and Linux system requirements. |
| HIP / hipCC | ROCm 7.x | Native C++ kernel language and compiler | Official CUDA-to-HIP porting path; closest replacement for CUDA C++ examples and build flow. |
| PyTorch for ROCm | ROCm 7.0 wheels or newer | Python tensor runtime and reference kernels | Existing project depends heavily on PyTorch; AMD docs provide ROCm-specific PyTorch install flow. |
| Triton for ROCm | Match ROCm/PyTorch support | Python DSL kernel path | AMD documents Triton for ROCm as a high-performance GPU programming route. |
| rocprofiler-sdk / rocprofv3 | ROCm 7.x | Profiling and trace collection | Replacement direction for CUPTI-based timing/profile analysis. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rocBLAS / hipBLASLt | ROCm 7.x | BLAS and GEMM backends | Replacement for cuBLAS-style GEMM paths and Level-3 BLAS. |
| MIOpen | ROCm 7.x | Deep learning primitives | Replacement candidate for cuDNN examples and softmax/normalization primitives where APIs fit. |
| Composable Kernel | ROCm 7.x | Templated high-performance ML kernels | Closest broad replacement candidate for CUTLASS/CuTe-style optimized kernels. |
| rocWMMA | ROCm 7.x | Matrix multiply-accumulate fragments | Use for lower-level MMA kernels where CK is too high-level. |
| hipCUB / rocPRIM / rocThrust | ROCm 7.x | Parallel primitives | Replacement for CUB/Thrust-style primitives in ported native kernels. |
| AMD SMI / ROCm SMI / rocminfo | ROCm 7.x | Hardware discovery and clock/health checks | Replacement direction for `nvidia-smi`-based discovery and clock logic. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| HIPIFY | CUDA-to-HIP source scanning and translation | Use `--examine` early to scope automatic vs manual porting. |
| amdclang++ / hipcc | Compile HIP/C++ code | rocWMMA docs prefer amdclang/amdclang++; hipcc is supported but may not be the future default. |
| rocprofv3 | Runtime/kernel profiling | Supports runtime/system/HSA traces and counter collection without source modifications. |
| ROCm Compute Profiler | Kernel-level performance analysis | Useful for phase-level profiling and comparing RDNA/CDNA behavior. |
| ROCgdb / ROCr Debug Agent | Debugging | Use for native HIP failures that cannot be diagnosed from Python logs. |

## Installation Direction

```bash
# Base direction, exact image/package names should be pinned during implementation.
# Replace CUDA image with official ROCm base image.
# Install ROCm >= 7.0, HIP dev tools, rocprofiler-sdk, rocBLAS, MIOpen, CK,
# rocWMMA, hipCUB/rocPRIM/rocThrust, and PyTorch ROCm wheels.

pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

The PyTorch command above is from AMD's ROCm 7.0 PyTorch installation docs and
should be revisited when pinning the Docker image.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| HIP native compilation | CUDA compatibility shims | Avoid for v1 because the project target is ROCm-only, not dual backend. |
| rocprofiler-sdk / rocprofv3 | CUDA event/CUPTI logic | Keep CUDA logic only as conceptual reference; ROCm implementation should use ROCm profiling APIs. |
| ROCm libraries | Reimplement all primitives manually | Manual HIP kernels are needed for examples, but library-backed paths should use optimized ROCm libraries first. |
| CK / rocWMMA | Direct assembly | Assembly can be future optimization work; v1 should favor maintainable, testable ROCm APIs. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `nvidia/cuda` Docker base | Pulls the wrong driver/toolchain assumptions | ROCm development/runtime image. |
| PyTorch CUDA wheels | Incompatible with ROCm runtime target | PyTorch ROCm wheels or ROCm PyTorch Docker image. |
| `nvidia-*` Python packages | CUDA/NVIDIA-specific dependencies | ROCm libraries and Python packages. |
| `nvidia-smi` clock/discovery logic | Not available on AMD GPUs | AMD SMI, ROCm SMI, `rocminfo`, and ROCm-specific clock APIs. |
| Blind HIPIFY output | HIPIFY cannot resolve all CUDA semantic differences | HIPIFY scan plus manual review and tests. |

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| ROCm 7.0.x | Ubuntu 24.04.x / 22.04.x and RHEL 9.x entries | Use official compatibility matrix when pinning Docker and CI. |
| PyTorch ROCm wheels | ROCm-specific wheel index | AMD docs show ROCm 7.0 wheel index; pin exact torch/torchvision after environment validation. |
| rocprofiler-sdk | ROCm 7.x | Use `rocprofv3` for runtime traces and counters. |
| RDNA 4 / CDNA 3 | ROCm compatibility matrix | Must be verified on actual target hardware, not assumed from code paths. |

## Sources

- AMD ROCm compatibility matrix — https://rocm.docs.amd.com/en/docs-7.0.1/compatibility/compatibility-matrix.html
- HIP porting guide — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
- PyTorch on ROCm install docs — https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.1/install/3rd-party/pytorch-install.html
- rocprofv3 docs — https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- Triton for ROCm docs — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-triton.html
- rocBLAS docs — https://rocmdocs.amd.com/projects/rocBLAS/en/develop/what-is-rocblas.html
- MIOpen docs — https://rocm.docs.amd.com/projects/MIOpen/en/latest/index.html
- Composable Kernel docs — https://rocmdocs.amd.com/projects/composable_kernel/en/develop/conceptual/Composable-Kernel-structure.html
- rocWMMA docs — https://rocmdocs.amd.com/projects/rocWMMA/en/develop/conceptual/programmers-guide.html

---
*Stack research for: ROCm port of SOL ExecBench*
*Researched: 2026-05-21*
