# Roadmap: SOL ExecBench ROCm Port

## Milestones

- [x] **v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration** - Phases 31-35, shipped 2026-05-22. See `.planning/milestones/v1.7-ROADMAP.md`.
- [x] **v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow** - Phases 27-30, shipped 2026-05-22. See `.planning/milestones/v1.6-ROADMAP.md`.
- [x] **v1.5 AMD-native SOL Scoring and ROCm Profiler Timing** - Phases 23-26, shipped 2026-05-22. See `.planning/milestones/v1.5-ROADMAP.md`.
- [x] **v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness** - Phases 19-22, shipped 2026-05-22. See `.planning/milestones/v1.4-ROADMAP.md`.
- [x] **v1.3 Non-CDNA Issue Closure** - Phases 14-18, shipped 2026-05-22. See `.planning/milestones/v1.3-ROADMAP.md`.
- [x] **v1.2 Engineering Practice Harvest and Compatibility Guardrails** - Phases 10-13, shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.
- [x] **v1.1 CDNA 3 Support and Migration Closure** - Phases 7-9, shipped 2026-05-21. See `.planning/milestones/v1.1-ROADMAP.md`.
- [x] **v1.0 ROCm Port** - Phases 1-6, shipped 2026-05-21. See `.planning/milestones/v1.0-ROADMAP.md`.

## Completed Phase History

<details>
<summary>v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration (Phases 31-35) - shipped 2026-05-22</summary>

- [x] Phase 31: Optimized Scoring Baseline Semantics (1/1 plan)
- [x] Phase 32: Source-Specific Profiler Timing Workflow (1/1 plan)
- [x] Phase 33: Reward-Hack Defense Expansion (1/1 plan)
- [x] Phase 34: ROCm Library Category Migration (1/1 plan)
- [x] Phase 35: MI300X Validation Readiness Guardrails (1/1 plan)

</details>

<details>
<summary>v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow (Phases 27-30) - shipped 2026-05-22</summary>

- [x] Phase 27: AMD SOL Analyzer Coverage (1/1 plan)
- [x] Phase 28: Live rocprofv3 Timing Integration (1/1 plan)
- [x] Phase 29: Derived AMD Scoring Workflow (1/1 plan)
- [x] Phase 30: Compatibility and Claim Guardrails (1/1 plan)

</details>

## Active Milestone

### v1.8 ROCm Library Ecosystem Completion

**Goal:** Promote remaining ROCm library replacement categories from candidate
or compatibility-only status to supported, runnable, tested public replacement
paths on RDNA 4.

**Hardware scope:** RDNA 4 validation only. CDNA 3 and CDNA 4 validation are
deferred and must not be claimed by this milestone.

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 36 | Library Build Plumbing and Diagnostics | Make dependency detection, compile metadata, Docker docs, and native staging ready for MIOpen, CK, and rocWMMA. | BUILD-01, BUILD-02, BUILD-03, BUILD-04 | 4 |
| 37 | MIOpen Supported Replacement | Promote MIOpen from candidate to supported RDNA 4 replacement with a real runnable example and tests. | MIOPEN-01, MIOPEN-02, MIOPEN-03, MIOPEN-04 | 4 |
| 38 | Composable Kernel Supported Replacement | Promote CK from candidate to supported RDNA 4 replacement for selected GEMM/fused GEMM workloads. | CK-01, CK-02, CK-03, CK-04 | 4 |
| 39 | rocWMMA Supported Replacement | Promote rocWMMA from candidate to supported RDNA 4 replacement for matrix-core GEMM-style workloads. | WMM-01, WMM-02, WMM-03, WMM-04 | 4 |
| 40 | Compatibility Cleanup and RDNA 4 Validation Closure | Remove support-status ambiguity, map former NVIDIA categories, and record RDNA 4-only completion evidence. | COMPAT-01, COMPAT-02, COMPAT-03, COMPAT-04, RDNA4-01, RDNA4-02, RDNA4-03 | 5 |

### Phase Details

**Phase 36: Library Build Plumbing and Diagnostics**

Goal: Make the native library infrastructure ready for MIOpen, CK, and rocWMMA
without changing public schema or trace contracts.

Requirements: BUILD-01, BUILD-02, BUILD-03, BUILD-04

Success criteria:
1. Dependency tests report MIOpen, CK, and rocWMMA missing headers/libraries by name.
2. Example compile metadata can express include paths, HIP flags, and linker flags through existing solution fields.
3. Docker/runtime docs identify required ROCm packages or installed files for all supported library examples.
4. Tests prove `miopen`, `ck`, and `rocwmma` use the native staging/compile flow.

**Phase 37: MIOpen Supported Replacement**

Goal: Promote MIOpen to supported status with a real RDNA 4 runnable public
example.

Requirements: MIOPEN-01, MIOPEN-02, MIOPEN-03, MIOPEN-04

Success criteria:
1. A public `examples/miopen/...` problem runs through `sol-execbench` on RDNA 4.
2. The solution source includes and calls MIOpen APIs for the measured implementation.
3. Tests cover MIOpen metadata, source consistency, dependency diagnostics, and RDNA 4 E2E behavior where hardware is present.
4. Docs identify MIOpen as the supported replacement for the former cuDNN softmax-style path with constraints.

**Phase 38: Composable Kernel Supported Replacement**

Goal: Promote CK to supported status for selected RDNA 4 GEMM or fused GEMM
workloads.

Requirements: CK-01, CK-02, CK-03, CK-04

Success criteria:
1. A public `examples/ck/...` problem runs through `sol-execbench` on RDNA 4.
2. The solution source uses real CK headers/API patterns for the measured implementation.
3. Tests cover CK metadata, source consistency, dependency diagnostics, and RDNA 4 E2E behavior where hardware is present.
4. Docs identify CK as a supported replacement for selected CUTLASS/CuTe-style workloads with scope limits.

**Phase 39: rocWMMA Supported Replacement**

Goal: Promote rocWMMA to supported status for RDNA 4 matrix-core GEMM-style
workloads.

Requirements: WMM-01, WMM-02, WMM-03, WMM-04

Success criteria:
1. A public `examples/rocwmma/...` problem runs through `sol-execbench` on RDNA 4.
2. The solution source uses real rocWMMA headers/API patterns for the measured implementation.
3. Tests cover rocWMMA metadata, source consistency, dependency diagnostics, and RDNA 4 E2E behavior where hardware is present.
4. Docs identify supported RDNA 4 targets and keep CDNA validation deferred.

**Phase 40: Compatibility Cleanup and RDNA 4 Validation Closure**

Goal: Close the library ecosystem gap by removing ambiguous compatibility
claims and recording RDNA 4 validation evidence.

Requirements: COMPAT-01, COMPAT-02, COMPAT-03, COMPAT-04, RDNA4-01, RDNA4-02,
RDNA4-03

Success criteria:
1. Public docs map former cuDNN, CUTLASS, CuTe DSL, and cuTile categories to supported, retired, or deferred ROCm statuses.
2. Compatibility examples no longer imply supported replacement status unless they contain real runnable library solutions.
3. Public-contract tests enforce support wording and RDNA 4-only validation scope.
4. Focused library example suite passes on RDNA 4 for hipBLAS, MIOpen, CK, and rocWMMA.
5. Closure artifacts summarize supported RDNA 4 library categories and defer CDNA 3/CDNA 4 validation.

## Current Planning State

Milestone v1.8 is initialized and ready for phase planning.

Next phase: Phase 36, Library Build Plumbing and Diagnostics.
