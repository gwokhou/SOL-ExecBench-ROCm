# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-21
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1 Requirements

### Environment

- [ ] **ENV-01**: Developer can build a ROCm >= 7.0 Docker image for this repository.
- [ ] **ENV-02**: Docker image includes ROCm runtime, HIP compiler tooling, ROCm profiling tools, AMD system management tools, and required ROCm libraries.
- [ ] **ENV-03**: Docker image installs PyTorch for ROCm and Triton for ROCm without CUDA wheel dependencies.
- [ ] **ENV-04**: Docker dependency tests verify ROCm, HIP, PyTorch ROCm, Triton ROCm, and selected ROCm libraries are importable or executable.

### Schema and Configuration

- [ ] **SCFG-01**: Solution language and hardware enums represent ROCm targets, including HIP/C++ and AMD architecture targets.
- [ ] **SCFG-02**: Existing problem schemas for `definition.json`, `workload.jsonl`, `solution.json`, and traces remain compatible unless a ROCm-specific change is explicitly documented.
- [ ] **SCFG-03**: NVIDIA/CUDA-only dependency declarations are removed or replaced with ROCm equivalents.
- [ ] **SCFG-04**: Documentation states that this port is ROCm-only and does not maintain CUDA/NVIDIA runtime support.

### Native Build

- [ ] **BUILD-01**: HIP/C++ solution sources can be staged, compiled, and loaded through the existing driver flow.
- [ ] **BUILD-02**: Native compile logic uses ROCm compiler/toolchain flags and AMD gfx target handling instead of CUDA `-gencode` flags.
- [ ] **BUILD-03**: Build failures produce actionable logs without corrupting trace JSON output.
- [ ] **BUILD-04**: HIPIFY or equivalent audit output is available to identify remaining CUDA-specific source patterns during the port.

### Evaluation Runtime

- [ ] **EVAL-01**: PyTorch ROCm solutions run through the eval driver and produce valid trace JSONL.
- [ ] **EVAL-02**: Triton ROCm solutions run through the eval driver and produce valid trace JSONL.
- [ ] **EVAL-03**: HIP/C++ shared-object solutions run through the eval driver and produce valid trace JSONL.
- [ ] **EVAL-04**: Destination-passing style and return-value style solution conventions continue to work under ROCm.
- [ ] **EVAL-05**: Input generation, output allocation, output normalization, and correctness checks preserve existing benchmark semantics under ROCm.

### Profiling and Timing

- [ ] **PROF-01**: Evaluation timing no longer depends on CUPTI or CUDA-only APIs.
- [ ] **PROF-02**: ROCm-native timing/profiling uses rocprofiler-sdk, rocprofv3, HIP events, or a validated combination that preserves benchmark integrity.
- [ ] **PROF-03**: Timing tests cover non-default stream or equivalent asynchronous-work hiding risks under ROCm.
- [ ] **PROF-04**: Trace environment data reports AMD GPU and ROCm library/tool versions.
- [ ] **PROF-05**: Clock and hardware checks use AMD SMI, ROCm SMI, `rocminfo`, or equivalent ROCm tooling instead of `nvidia-smi`.

### Library and Example Migration

- [ ] **LIB-01**: PyTorch examples are migrated to ROCm-compatible execution and pass adapted tests.
- [ ] **LIB-02**: Triton examples are migrated to Triton ROCm-compatible execution and pass adapted tests.
- [ ] **LIB-03**: CUDA C++ examples are migrated to HIP/C++ and pass adapted tests.
- [ ] **LIB-04**: CUTLASS-style examples are replaced or reimplemented using ROCm-appropriate libraries or HIP kernels where feasible.
- [ ] **LIB-05**: cuDNN-style examples are replaced or reimplemented using MIOpen or ROCm-appropriate kernels where feasible.
- [ ] **LIB-06**: CuTe DSL and cuTile-style examples are replaced, reimplemented, or explicitly documented with ROCm alternatives where feasible.
- [ ] **LIB-07**: Replacement decisions for rocBLAS, hipBLASLt, MIOpen, Composable Kernel, rocWMMA, hipCUB, rocPRIM, and rocThrust are documented.

### Testing and Validation

- [ ] **TEST-01**: Existing pytest suite is migrated from CUDA/NVIDIA assumptions to ROCm/AMD assumptions.
- [ ] **TEST-02**: Hardware markers distinguish unavailable ROCm, unsupported architecture, RDNA 4, and CDNA 3 cases.
- [ ] **TEST-03**: Adapted unit, driver, benchmark-helper, Docker dependency, example, and e2e tests pass in a ROCm >= 7.0 environment.
- [ ] **TEST-04**: Adapted test suite passes on at least one RDNA 4 GPU environment.
- [ ] **TEST-05**: Adapted test suite passes on at least one CDNA 3 GPU environment.
- [ ] **TEST-06**: Reward-hack defense tests remain active and pass under ROCm.

### Documentation and Compliance

- [ ] **DOC-01**: README and setup docs explain ROCm installation, Docker usage, dataset setup, and local evaluation commands.
- [ ] **DOC-02**: Schema docs describe ROCm-supported languages, hardware targets, and replacement limitations.
- [ ] **DOC-03**: Profiling/analyze docs explain the ROCm-native tooling path.
- [ ] **DOC-04**: License and third-party notices are reviewed and updated for all retained and replacement dependencies.
- [ ] **DOC-05**: Known gaps or unsupported NVIDIA-equivalent features are documented clearly.

## v2 Requirements

### Future Expansion

- **FUT-01**: Optional dual CUDA/ROCm backend support can be considered if upstream parity becomes a separate product goal.
- **FUT-02**: Additional AMD architectures beyond RDNA 4 and CDNA 3 can be added after v1 validation.
- **FUT-03**: Assembly-level AMD kernel optimization can be explored after semantic correctness and test coverage are stable.
- **FUT-04**: Extended profile report visualization can be added after core profiling data is reliable.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Maintaining CUDA/NVIDIA runtime support | User selected a ROCm-only port and allowed NVIDIA paths to be removed. |
| Separate NVIDIA/ROCm parity test suite for v1 | User selected adapted existing tests as the completion standard. |
| Guaranteeing perfect one-to-one replacements for every NVIDIA DSL | Some CUDA/CUTLASS/cuDNN/CuTe/cuTile capabilities may require ROCm-appropriate alternatives or documented limitations. |
| Supporting ROCm versions below 7.0 | User target is ROCm >= 7.0. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Pending |
| ENV-02 | Phase 1 | Pending |
| ENV-03 | Phase 1 | Pending |
| ENV-04 | Phase 1 | Pending |
| SCFG-01 | Phase 2 | Pending |
| SCFG-02 | Phase 2 | Pending |
| SCFG-03 | Phase 1 | Pending |
| SCFG-04 | Phase 6 | Pending |
| BUILD-01 | Phase 2 | Pending |
| BUILD-02 | Phase 2 | Pending |
| BUILD-03 | Phase 2 | Pending |
| BUILD-04 | Phase 2 | Pending |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Pending |
| EVAL-03 | Phase 3 | Pending |
| EVAL-04 | Phase 3 | Pending |
| EVAL-05 | Phase 3 | Pending |
| PROF-01 | Phase 3 | Pending |
| PROF-02 | Phase 3 | Pending |
| PROF-03 | Phase 3 | Pending |
| PROF-04 | Phase 3 | Pending |
| PROF-05 | Phase 3 | Pending |
| LIB-01 | Phase 4 | Pending |
| LIB-02 | Phase 4 | Pending |
| LIB-03 | Phase 4 | Pending |
| LIB-04 | Phase 4 | Pending |
| LIB-05 | Phase 4 | Pending |
| LIB-06 | Phase 4 | Pending |
| LIB-07 | Phase 4 | Pending |
| TEST-01 | Phase 5 | Pending |
| TEST-02 | Phase 5 | Pending |
| TEST-03 | Phase 5 | Pending |
| TEST-04 | Phase 5 | Pending |
| TEST-05 | Phase 5 | Pending |
| TEST-06 | Phase 5 | Pending |
| DOC-01 | Phase 6 | Pending |
| DOC-02 | Phase 6 | Pending |
| DOC-03 | Phase 6 | Pending |
| DOC-04 | Phase 6 | Pending |
| DOC-05 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 39 total
- Mapped to phases: 39
- Unmapped: 0

---
*Requirements defined: 2026-05-21*
*Last updated: 2026-05-21 after initialization*
