# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-22
**Milestone:** v1.8 ROCm Library Ecosystem Completion
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1.8 Requirements

### Library Build Plumbing

- [x] **BUILD-01**: Maintainer can verify MIOpen, Composable Kernel, and rocWMMA dependency availability with clear missing-header or missing-library diagnostics.
- [x] **BUILD-02**: Native library solutions can express required include, compiler, and linker flags through existing solution compile metadata without changing public schema fields.
- [x] **BUILD-03**: Docker/runtime documentation lists the ROCm packages or files required for MIOpen, Composable Kernel, rocWMMA, and hipBLAS examples.
- [x] **BUILD-04**: Build and packaging tests prove `miopen`, `ck`, and `rocwmma` solutions stage through the native ROCm compile path rather than compatibility-only Python paths.

### MIOpen Replacement

- [x] **MIOPEN-01**: User can run a public MIOpen-backed example on RDNA 4 through `sol-execbench` and receive passing trace JSONL.
- [x] **MIOPEN-02**: The MIOpen example uses real MIOpen headers/API calls and does not silently fall back to PyTorch for the measured implementation.
- [x] **MIOPEN-03**: Tests cover MIOpen solution metadata, source consistency, dependency detection, and RDNA 4 E2E behavior where hardware is available.
- [x] **MIOPEN-04**: Documentation identifies MIOpen as the supported ROCm replacement path for the former cuDNN softmax-style example, including operation-specific constraints.

### Composable Kernel Replacement

- [x] **CK-01**: User can run a public Composable Kernel-backed example on RDNA 4 through `sol-execbench` and receive passing trace JSONL.
- [x] **CK-02**: The CK example uses real CK headers/API patterns and does not silently fall back to PyTorch for the measured implementation.
- [x] **CK-03**: Tests cover CK solution metadata, source consistency, dependency detection, and RDNA 4 E2E behavior where hardware is available.
- [x] **CK-04**: Documentation identifies CK as a supported ROCm replacement path for selected CUTLASS/CuTe-style GEMM or fused GEMM workloads, including known scope limits.

### rocWMMA Replacement

- [ ] **WMM-01**: User can run a public rocWMMA-backed example on RDNA 4 through `sol-execbench` and receive passing trace JSONL.
- [ ] **WMM-02**: The rocWMMA example uses real rocWMMA headers/API patterns and does not silently fall back to PyTorch for the measured implementation.
- [ ] **WMM-03**: Tests cover rocWMMA solution metadata, source consistency, dependency detection, and RDNA 4 E2E behavior where hardware is available.
- [ ] **WMM-04**: Documentation identifies rocWMMA as a supported ROCm replacement path for matrix-core GEMM-style workloads on supported RDNA 4 targets, with CDNA validation deferred.

### Compatibility Cleanup and Claims

- [ ] **COMPAT-01**: Public docs map each former NVIDIA library/DSL category to a supported ROCm replacement, a retired compatibility example, or a clearly deferred validation target.
- [ ] **COMPAT-02**: Former cuDNN, CUTLASS, CuTe DSL, and cuTile example paths no longer imply supported library replacement unless they contain a real runnable ROCm library solution.
- [ ] **COMPAT-03**: Public-contract tests prevent MIOpen, CK, or rocWMMA from being documented as supported unless runnable examples and tests exist.
- [ ] **COMPAT-04**: README and support docs state that v1.8 validation is RDNA 4 only and that CDNA 3 and CDNA 4 validation are deferred.

### RDNA 4 Validation Closure

- [ ] **RDNA4-01**: Maintainer can run the focused library example suite on RDNA 4 and record passing evidence for hipBLAS, MIOpen, CK, and rocWMMA examples.
- [ ] **RDNA4-02**: Maintainer can run focused unit/docs tests that protect dependency diagnostics, native staging, solution metadata, and support-status wording.
- [ ] **RDNA4-03**: Completion artifacts summarize which library categories are supported on RDNA 4 and explicitly avoid CDNA 3/CDNA 4 validation claims.

## Future Requirements

### Hardware Validation

- **HW-01**: Run the full adapted suite on AMD MI300X/CDNA3 and record evidence before claiming commercial GPU hardware validation.
- **HW-02**: Validate FP8 behavior and performance on MI300X once hardware access is available.
- **HW-03**: Validate ROCm library replacement examples on CDNA 3 and CDNA 4 when appropriate hardware access is available.

### Performance and Paper-Parity Work

- **PERF-01**: Add profiler-backed performance comparison reports for supported ROCm library examples.
- **DATA-01**: Recreate or adapt the original paper's model-to-subgraph extraction and curation pipeline.
- **SOLAR-01**: Continue toward deeper upstream SOLAR parity with graph tracing, einsum/IR conversion, lookup validation, and tighter movement bounds.
- **MXFP4-01**: Validate NVFP4/MXFP4-like paths only when suitable AMD hardware support and methodology exist.

## Out of Scope

| Feature | Reason |
|---------|--------|
| CDNA 3 library validation in v1.8 | User explicitly scoped this milestone to RDNA 4 validation only. |
| CDNA 4 library validation in v1.8 | User explicitly deferred CDNA 4 validation. |
| NVIDIA/B200 runtime parity | The project is ROCm-only; original NVIDIA hardware is not a missing feature. |
| Full original dataset extraction pipeline | This milestone focuses on ROCm library replacement support, not paper data generation. |
| Full upstream SOLAR parity | This milestone focuses on library ecosystem completeness. |
| Peak performance claims for library examples | Runnable support and correctness come first; performance claims require separate profiler and baseline evidence. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUILD-01 | Phase 36 | Complete |
| BUILD-02 | Phase 36 | Complete |
| BUILD-03 | Phase 36 | Complete |
| BUILD-04 | Phase 36 | Complete |
| MIOPEN-01 | Phase 37 | Complete |
| MIOPEN-02 | Phase 37 | Complete |
| MIOPEN-03 | Phase 37 | Complete |
| MIOPEN-04 | Phase 37 | Complete |
| CK-01 | Phase 38 | Complete |
| CK-02 | Phase 38 | Complete |
| CK-03 | Phase 38 | Complete |
| CK-04 | Phase 38 | Complete |
| WMM-01 | Phase 39 | Pending |
| WMM-02 | Phase 39 | Pending |
| WMM-03 | Phase 39 | Pending |
| WMM-04 | Phase 39 | Pending |
| COMPAT-01 | Phase 40 | Pending |
| COMPAT-02 | Phase 40 | Pending |
| COMPAT-03 | Phase 40 | Pending |
| COMPAT-04 | Phase 40 | Pending |
| RDNA4-01 | Phase 40 | Pending |
| RDNA4-02 | Phase 40 | Pending |
| RDNA4-03 | Phase 40 | Pending |

**Coverage:**
- v1.8 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0

---
*Requirements defined: 2026-05-22*
*Last updated: 2026-05-22 after Phase 38 completion*
