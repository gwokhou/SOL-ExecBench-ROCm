# Feature Research

**Domain:** ROCm port of GPU kernel benchmark framework
**Researched:** 2026-05-21
**Confidence:** HIGH for table-stakes capabilities, MEDIUM for complete DSL/library equivalence

## Feature Landscape

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| ROCm Docker environment | Users need reproducible ROCm 7+ setup | HIGH | Replace CUDA base image, packages, env vars, SMI tooling, and dependency checks. |
| HIP/C++ solution compilation | CUDA C++ is a core original path | HIGH | Port `build_ext.py`, language enums, examples, and architecture flags. |
| PyTorch ROCm evaluation | Existing reference and solution code depends on torch | MEDIUM | Swap CUDA wheels and device assumptions while preserving schemas. |
| Triton on ROCm support | Original benchmark includes Triton kernels | MEDIUM | Validate Triton ROCm install and example compatibility. |
| ROCm profiling/timing | Benchmark integrity depends on reliable GPU time | HIGH | Replace CUPTI timing with rocprofiler-sdk/rocprofv3 or robust HIP event strategy. |
| AMD hardware detection | Tests must target RDNA 4 and CDNA 3 | MEDIUM | Replace `nvidia-smi` and SM markers with AMD architecture detection. |
| Migrated tests | Done requires adapted test suite passing | HIGH | Convert Docker dependency, e2e, timing, and examples tests. |
| License audit | User explicitly requires LICENSE compliance | MEDIUM | Track replacement dependency licenses and retained NVIDIA-origin code. |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Broad original-language replacement matrix | Makes the ROCm port useful for SOL-style LLM kernel evaluation | HIGH | Map CUDA/CUTLASS/cuDNN/CuTe/cuTile to HIP, CK, MIOpen, rocWMMA, rocPRIM, etc. |
| RDNA 4 plus CDNA 3 validation | Covers consumer/workstation and datacenter AMD GPU families | HIGH | Needs hardware-specific skip/xfail strategy and real runs. |
| ROCm-native reward-hack defenses | Preserves benchmark trustworthiness | HIGH | Stream/timing/lazy-output checks must be revalidated under ROCm. |
| Porting diagnostics | Helps developers understand unsupported CUDA idioms | MEDIUM | Integrate HIPIFY examine output and clear failure messages. |

### Anti-Features

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Dual CUDA + ROCm backend | Appears to preserve upstream compatibility | Doubles test matrix and conflicts with user's ROCm-only target | Remove or isolate NVIDIA paths and document ROCm-only support. |
| One-shot HIPIFY conversion | Fast apparent progress | Misses semantic issues like warp size, masks, libraries, profiling, and architecture flags | Use HIPIFY for inventory, then manually port by subsystem. |
| Treating HIP events as equivalent to CUPTI | Simple timing replacement | May miss non-default stream or profiler-level measurement issues | Validate rocprofiler-sdk/rocprofv3 timing behavior before accepting. |
| Keeping NVIDIA package names in schemas | Minimizes model changes | Confuses users and hides ROCm semantics | Introduce ROCm language/library names while preserving JSON compatibility where possible. |

## Feature Dependencies

```text
ROCm Docker baseline
  -> PyTorch ROCm install
  -> HIP/C++ compile toolchain
  -> Docker dependency tests

HIP/C++ compile toolchain
  -> HIP examples
  -> native e2e tests

ROCm profiling/timing
  -> benchmark correctness/performance traces
  -> reward-hack validation

Library replacement matrix
  -> example migration
  -> adapted test suite
  -> RDNA4/CDNA3 validation
```

## MVP Definition

### Launch With (v1)

- [ ] ROCm 7+ Docker/dev environment.
- [ ] PyTorch ROCm, HIP/C++, and Triton ROCm evaluation paths.
- [ ] Replacement strategy and implementation for original NVIDIA library/DSL categories.
- [ ] ROCm-native profiling/timing and hardware detection.
- [ ] Migrated examples and tests passing on RDNA 4 and CDNA 3.
- [ ] License/dependency compliance notes.

### Add After Validation (v1.x)

- [ ] More optimized AMD kernels after semantic correctness is stable.
- [ ] Better profile report export and analysis tooling.
- [ ] Additional AMD architectures beyond RDNA 4 and CDNA 3.

### Future Consideration (v2+)

- [ ] Optional dual backend support if upstream parity becomes a separate product goal.
- [ ] Deeper assembly-level or framework-specific AMD DSL exploration.

## Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| ROCm Docker baseline | HIGH | HIGH | P1 |
| HIP/C++ compile path | HIGH | HIGH | P1 |
| PyTorch ROCm path | HIGH | MEDIUM | P1 |
| ROCm profiling/timing | HIGH | HIGH | P1 |
| Migrated tests | HIGH | HIGH | P1 |
| Full replacement matrix | HIGH | HIGH | P1 |
| Porting diagnostics | MEDIUM | MEDIUM | P2 |
| Extra optimized kernels | MEDIUM | HIGH | P2 |

## Sources

- AMD ROCm compatibility matrix — ROCm 7.0.x system and component baseline.
- HIP porting guide and HIPIFY docs — CUDA-to-HIP migration workflow and known semantic traps.
- PyTorch ROCm install docs — ROCm wheel setup and validation expectations.
- rocprofv3 docs — profiling and tracing capabilities.
- ROCm library docs — rocBLAS, MIOpen, Composable Kernel, rocWMMA, rocPRIM, hipCUB.

---
*Feature research for: ROCm port of SOL ExecBench*
*Researched: 2026-05-21*
