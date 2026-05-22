# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-22
**Milestone:** v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1.7 Requirements

### Optimized Scoring Baselines

- [x] **BASE-01**: Maintainers can store optimized scoring baseline timing artifacts separately from PyTorch reference timing and canonical trace JSONL.
- [x] **BASE-02**: AMD-native score reports use an explicit scoring baseline input when available, and label fallback-to-reference scoring as provisional rather than release-defined.
- [x] **BASE-03**: Dataset runs can consume baseline artifacts by definition and workload UUID without mutating existing trace records.
- [x] **BASE-04**: Baseline comparison and AMD-native score documentation explain reference timing, candidate timing, scoring baseline timing, and AMD SOL bound roles.

### Source-Specific Timing Workflow

- [x] **TIME-01**: HIP native, Triton, and PyTorch/operator sources have explicit timing backend selection that records whether timing came from `rocprofv3` or event fallback.
- [x] **TIME-02**: The evaluation or dataset workflow can invoke `rocprofv3` collection end-to-end for profiler-backed timing evidence when the selected source policy supports it.
- [x] **TIME-03**: Real or fixture-backed `rocprofv3` parser coverage validates representative kernel, HIP runtime, missing-output, and command-failure cases.
- [x] **TIME-04**: Timing reports preserve aggregation rule, trial count, warmup count, iteration count, clock-lock status, hardware architecture, and fallback reason.

### Reward-Hack Hardening

- [ ] **HACK-01**: Evaluation rejects or flags hidden-work patterns using non-default streams or stream-like asynchronous dispatch that can evade the selected timer.
- [ ] **HACK-02**: Evaluation rejects or flags semantic output caching across correctness and timing phases, including data-pointer and content-keyed cache patterns where practical.
- [ ] **HACK-03**: Evaluation rejects unauthorized file I/O, embedded opaque binary payloads, base64-loaded code objects, and runtime dynamic native loading outside approved build paths.
- [ ] **HACK-04**: Evaluation rejects precision downgrade abuse when output dtype alone is insufficient to prove numerical contract preservation.
- [ ] **HACK-05**: A static submission review layer reports suspicious Python and native-source patterns before execution without blocking legitimate HIP/Triton/PyTorch solutions by default.

### ROCm Library Migration

- [ ] **LIB-01**: `hipblas` or hipBLASLt has at least one runnable public example, build path, and focused test that demonstrates a supported ROCm BLAS-backed solution category.
- [ ] **LIB-02**: MIOpen has at least one runnable or clearly guarded public example and test path, or is explicitly retained as candidate with a concrete blocker.
- [ ] **LIB-03**: Composable Kernel has at least one runnable or clearly guarded public example and test path, or is explicitly retained as candidate with a concrete blocker.
- [ ] **LIB-04**: rocWMMA has at least one runnable or clearly guarded public example and test path, or is explicitly retained as candidate with a concrete blocker.
- [ ] **LIB-05**: Documentation and public-contract tests accurately distinguish supported, guarded, and candidate ROCm library categories.

### MI300X Validation Readiness

- [ ] **MI3-01**: MI300X/CDNA3 validation instructions identify required hardware, ROCm version, clock-lock setup, commands, artifacts, and acceptance criteria.
- [ ] **MI3-02**: FP8 validation readiness is documented for MI300X while NVFP4/MXFP4 remains explicitly deferred until suitable AMD hardware support exists.
- [ ] **MI3-03**: Reports cannot upgrade CDNA3/MI300X validation status unless a full adapted suite run and environment evidence are recorded.

## Future Requirements

### Hardware Validation

- **HW-01**: Run the full adapted suite on AMD MI300X/CDNA3 and record evidence before claiming commercial GPU hardware validation.
- **HW-02**: Validate FP8 behavior and performance on MI300X once hardware access is available.

### Deferred Paper-Parity Work

- **DATA-01**: Recreate or adapt the original paper's model-to-subgraph extraction and curation pipeline.
- **SOLAR-01**: Continue toward deeper upstream SOLAR parity with graph tracing, einsum/IR conversion, lookup validation, and tighter movement bounds.
- **MXFP4-01**: Validate NVFP4/MXFP4-like paths only when suitable AMD hardware support and methodology exist.

## Out of Scope

| Feature | Reason |
|---------|--------|
| NVIDIA/B200 runtime parity | The project is ROCm-only; original NVIDIA hardware is no longer treated as a missing feature. |
| Full original dataset extraction pipeline in v1.7 | Explicitly deferred by user decision. |
| Full upstream SOLAR parity in v1.7 | Explicitly deferred; v1.7 focuses on evaluation credibility and library migration. |
| MI300X full-suite execution in v1.7 without hardware | Validation readiness is in scope; the real run waits for hardware access. |
| NVFP4/MXFP4 hardware validation | No suitable AMD hardware validation path is available now. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BASE-01 | Phase 31 | Complete |
| BASE-02 | Phase 31 | Complete |
| BASE-03 | Phase 31 | Complete |
| BASE-04 | Phase 31 | Complete |
| TIME-01 | Phase 32 | Complete |
| TIME-02 | Phase 32 | Complete |
| TIME-03 | Phase 32 | Complete |
| TIME-04 | Phase 32 | Complete |
| HACK-01 | Phase 33 | Pending |
| HACK-02 | Phase 33 | Pending |
| HACK-03 | Phase 33 | Pending |
| HACK-04 | Phase 33 | Pending |
| HACK-05 | Phase 33 | Pending |
| LIB-01 | Phase 34 | Pending |
| LIB-02 | Phase 34 | Pending |
| LIB-03 | Phase 34 | Pending |
| LIB-04 | Phase 34 | Pending |
| LIB-05 | Phase 34 | Pending |
| MI3-01 | Phase 35 | Pending |
| MI3-02 | Phase 35 | Pending |
| MI3-03 | Phase 35 | Pending |

**Coverage:**
- v1.7 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---
*Requirements defined: 2026-05-22*
*Last updated: 2026-05-22 after v1.7 milestone initialization*
