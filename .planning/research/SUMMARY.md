# Project Research Summary

**Project:** SOL ExecBench ROCm Port
**Domain:** ROCm port of GPU kernel benchmark framework
**Researched:** 2026-05-21
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project should be treated as a backend replacement of a benchmark runtime,
not as a simple CUDA-to-HIP search/replace. The existing CLI, JSON schemas,
staging model, trace format, correctness logic, and test organization are worth
preserving. The NVIDIA-specific layers are Docker/runtime dependencies, native
compile flags, eval-driver CUDA assumptions, CUPTI/timing/profile logic,
hardware discovery/clock controls, examples, and Docker dependency checks.

The recommended ROCm stack is ROCm >= 7.0 with HIP/hipCC or amdclang++ for native
kernels, PyTorch ROCm wheels or ROCm PyTorch images for tensor execution,
Triton ROCm for Python DSL kernels, rocprofiler-sdk/rocprofv3 for profiling,
AMD SMI/ROCm SMI/rocminfo for system inspection, and ROCm libraries such as
rocBLAS/hipBLASLt, MIOpen, Composable Kernel, rocWMMA, hipCUB, rocPRIM, and
rocThrust for replacements of NVIDIA library/DSL examples.

The main risk is benchmark drift: a port that compiles and runs but no longer
measures the same thing or no longer defends against the same reward-hacking
patterns. The roadmap should therefore stabilize environment, build, eval, and
timing contracts before broad example migration.

## Key Findings

### Recommended Stack

**Core technologies:**
- ROCm >= 7.0: required platform baseline.
- HIP/hipCC or amdclang++: native kernel compile path.
- PyTorch ROCm: replacement for current PyTorch CUDA dependency.
- Triton ROCm: replacement for Triton examples where supported.
- rocprofiler-sdk/rocprofv3: replacement direction for CUPTI profiling/timing.
- ROCm libraries: rocBLAS/hipBLASLt, MIOpen, Composable Kernel, rocWMMA, hipCUB, rocPRIM, rocThrust.

### Expected Features

**Must have:**
- ROCm Docker/dev environment.
- HIP/C++ native compile and load path.
- PyTorch ROCm and Triton ROCm solution paths.
- ROCm-native profiling/timing and hardware detection.
- Replacement matrix for original CUDA/CUTLASS/cuDNN/CuTe/cuTile capabilities.
- Adapted tests passing on RDNA 4 and CDNA 3.
- License and third-party dependency compliance.

**Should have:**
- Porting diagnostics using HIPIFY examine output.
- ROCm-native profile/analyze workflow documentation.
- Clear architecture-specific skip/failure reporting.

**Defer:**
- Dual CUDA/ROCm backend.
- Assembly-level AMD-specific optimization beyond what is needed for correctness and representative examples.

### Architecture Approach

Keep the current package and data contracts. Replace NVIDIA-specific behavior
inside existing backend boundaries: Docker/dependencies, `ProblemPackager`,
`build_ext.py`, `eval_driver.py`, timing/profile helpers, clock/hardware helpers,
examples, and tests. This limits blast radius and keeps the ROCm port aligned
with the original benchmark semantics.

**Major components:**
1. Environment baseline — Docker, dependencies, ROCm package validation.
2. Native build path — HIP language/schema support and gfx target compilation.
3. Eval runtime — ROCm-aware import, input/output, correctness, timing, reward-hack checks.
4. Library/example migration — ROCm equivalents for original solution categories.
5. Validation — adapted test suite on RDNA 4 and CDNA 3.

### Critical Pitfalls

1. **Assuming HIPIFY completes the port** — use it to scope, then manually review semantic differences.
2. **Replacing CUPTI timing weakly** — validate ROCm profiler/timing against stream and reward-hack tests.
3. **Migrating examples too early** — stabilize backend contracts first.
4. **Treating RDNA 4 and CDNA 3 as equivalent** — validate both explicitly.
5. **License drift** — review replacement libraries and copied code continuously.

## Implications for Roadmap

### Phase 1: ROCm Environment Baseline
**Rationale:** Everything depends on a reproducible ROCm 7+ environment.
**Delivers:** Docker image, dependency installation, ROCm PyTorch/Triton/HIP checks, AMD hardware discovery smoke tests.
**Addresses:** Docker, package, and system-tool replacement.
**Avoids:** Environment drift hiding as code failure.

### Phase 2: Schema and Native Build Port
**Rationale:** Native solutions need ROCm language/hardware concepts before examples can migrate.
**Delivers:** HIP language/hardware support, gfx target handling, HIP build template, replacement for CUDA compile flags.
**Uses:** HIP, hipCC/amdclang++, HIPIFY inventory.

### Phase 3: ROCm Eval Driver and Timing
**Rationale:** Benchmark validity depends on isolated execution and trustworthy timing.
**Delivers:** ROCm-aware eval driver, PyTorch/Triton ROCm runtime behavior, rocprofiler/HIP timing path, clock/environment capture.
**Avoids:** Benchmark drift and reward-hack regressions.

### Phase 4: Library and Example Migration
**Rationale:** Once runtime contracts are stable, port examples by category.
**Delivers:** HIP/PyTorch/Triton examples plus ROCm replacements for CUTLASS/cuDNN/CuTe/cuTile categories where feasible.
**Uses:** rocBLAS/hipBLASLt, MIOpen, Composable Kernel, rocWMMA, rocPRIM/hipCUB/rocThrust.

### Phase 5: Test Suite Adaptation and Hardware Validation
**Rationale:** User-defined done is adapted tests passing on ROCm >= 7.0 with RDNA 4 and CDNA 3.
**Delivers:** Migrated pytest markers, Docker dependency tests, e2e examples, timing/reward-hack tests, recorded hardware validation.

### Phase 6: Documentation and License Compliance
**Rationale:** Researchers and developers need clear usage and compliance boundaries.
**Delivers:** README/docs updates, solution schema docs, Docker instructions, profile/analyze docs, third-party notices review.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official ROCm docs identify core tools and libraries. |
| Features | MEDIUM-HIGH | User goal is clear; exact equivalents for every NVIDIA DSL/library require implementation validation. |
| Architecture | MEDIUM-HIGH | Existing code boundaries are clear; profiler/timing design needs proof. |
| Pitfalls | HIGH | Porting and benchmark-integrity risks are well-established. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- Exact ROCm replacements for CuTe DSL/cuTile examples need validation during planning.
- rocprofiler-sdk integration shape in Python subprocess needs a prototype.
- RDNA 4 support details must be verified on the actual target hardware and ROCm 7.x image.
- License compatibility must be checked when concrete replacement dependencies are selected.

## Sources

### Primary
- ROCm compatibility matrix — https://rocm.docs.amd.com/en/docs-7.0.1/compatibility/compatibility-matrix.html
- HIP porting guide — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
- PyTorch ROCm install docs — https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.1/install/3rd-party/pytorch-install.html
- rocprofv3 docs — https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- Triton for ROCm docs — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-triton.html
- ROCm library docs for rocBLAS, MIOpen, Composable Kernel, rocWMMA, and primitives.

### Local
- `.planning/PROJECT.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/STACK.md`
- `.planning/codebase/CONCERNS.md`

---
*Research completed: 2026-05-21*
*Ready for roadmap: yes*
