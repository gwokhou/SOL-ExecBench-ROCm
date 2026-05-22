# Roadmap: SOL ExecBench ROCm Port

## Current Milestone: v1.8 ROCm Library Ecosystem Completion

**Goal:** Promote remaining ROCm library replacement categories from candidate
or compatibility-only status to supported, runnable, tested public replacement
paths on RDNA 4.

**Hardware scope:** RDNA 4 validation only. CDNA 3 and CDNA 4 validation are
deferred and must not be claimed by this milestone.

**Phase numbering:** Continues from v1.7. v1.8 starts at Phase 36.

## Phase Summary

- [x] Phase 36: Library Build Plumbing and Diagnostics
- [x] Phase 37: MIOpen Supported Replacement
- [x] Phase 38: Composable Kernel Supported Replacement
- [x] Phase 39: rocWMMA Supported Replacement
- [x] Phase 40: Compatibility Cleanup and RDNA 4 Validation Closure

| Phase | Name | Goal | Requirements |
|-------|------|------|--------------|
| 36 | Library Build Plumbing and Diagnostics | Complete 2026-05-22: Added reusable ROCm library dependency diagnostics, Docker dependency checks, native staging tests, and dependency docs for MIOpen, CK, and rocWMMA. | BUILD-01, BUILD-02, BUILD-03, BUILD-04 |
| 37 | MIOpen Supported Replacement | Complete 2026-05-22: Added a native MIOpen softmax example, RDNA 4 E2E registration, source/staging tests, and supported-status docs. | MIOPEN-01, MIOPEN-02, MIOPEN-03, MIOPEN-04 |
| 38 | Composable Kernel Supported Replacement | Complete 2026-05-22: Added a CK-facing small GEMM example, RDNA 4 E2E registration, source/staging tests, and supported-status docs. | CK-01, CK-02, CK-03, CK-04 |
| 39 | rocWMMA Supported Replacement | Complete 2026-05-22: Added a rocWMMA matrix-core GEMM example, RDNA 4 E2E registration, source/staging tests, and supported-status docs. | WMM-01, WMM-02, WMM-03, WMM-04 |
| 40 | Compatibility Cleanup and RDNA 4 Validation Closure | Complete 2026-05-22: Cleaned public support wording, mapped compatibility paths, protected RDNA 4-only claims, and recorded focused validation evidence. | COMPAT-01, COMPAT-02, COMPAT-03, COMPAT-04, RDNA4-01, RDNA4-02, RDNA4-03 |

**Coverage:** 23 / 23 v1.8 requirements mapped. 0 unmapped.

## Phases

### Phase 36: Library Build Plumbing and Diagnostics

**Status:** Complete 2026-05-22

**Goal:** Make the native library infrastructure ready for MIOpen, CK, and
rocWMMA without changing public schema or trace contracts.

**Requirements:** BUILD-01, BUILD-02, BUILD-03, BUILD-04

**Success criteria:**

1. Dependency tests report MIOpen, CK, and rocWMMA missing headers/libraries by name.
2. Example compile metadata can express include paths, HIP flags, and linker flags through existing solution fields.
3. Docker/runtime docs identify required ROCm packages or installed files for all supported library examples.
4. Tests prove `miopen`, `ck`, and `rocwmma` use the native staging/compile flow.

**Implementation notes:**

- Preserve public solution schema and trace JSONL.
- Prefer existing `compile_options` fields before adding internal helpers.
- Keep RDNA 4 as the validation target for this milestone.

**Plans:** 1 plan

Plans:

- [x] 36-01: Add ROCm library dependency diagnostics, native staging tests,
  dependency docs, and focused verification for remaining candidate categories.

### Phase 37: MIOpen Supported Replacement

**Status:** Complete 2026-05-22

**Goal:** Promote MIOpen to supported status with a real RDNA 4 runnable public
example.

**Requirements:** MIOPEN-01, MIOPEN-02, MIOPEN-03, MIOPEN-04

**Success criteria:**

1. A public `examples/miopen/...` problem runs through `sol-execbench` on RDNA 4.
2. The solution source includes and calls MIOpen APIs for the measured implementation.
3. Tests cover MIOpen metadata, source consistency, dependency diagnostics, and RDNA 4 E2E behavior where hardware is present.
4. Docs identify MIOpen as the supported replacement for the former cuDNN softmax-style path with constraints.

**Implementation notes:**

- Softmax is the preferred first MIOpen example because the repo already has a
  former cuDNN softmax compatibility example.
- Do not silently fall back to PyTorch in the measured implementation.
- Document operation-specific MIOpen descriptor limitations.

**Plans:** 1 plan

Plans:

- [x] 37-01: Add MIOpen softmax supported example, metadata/source tests,
  native staging coverage, RDNA 4 E2E registration, and support docs.

### Phase 38: Composable Kernel Supported Replacement

**Status:** Complete 2026-05-22

**Goal:** Promote CK to supported status for selected RDNA 4 GEMM or fused GEMM
workloads.

**Requirements:** CK-01, CK-02, CK-03, CK-04

**Success criteria:**

1. A public `examples/ck/...` problem runs through `sol-execbench` on RDNA 4.
2. The solution source uses real CK headers/API patterns for the measured implementation.
3. Tests cover CK metadata, source consistency, dependency diagnostics, and RDNA 4 E2E behavior where hardware is present.
4. Docs identify CK as a supported replacement for selected CUTLASS/CuTe-style workloads with scope limits.

**Implementation notes:**

- Prefer a minimal GEMM or GEMM-with-epilogue example over a broad untested CK
  integration.
- Use existing native packaging conventions where possible.
- Keep support claims tied to runnable example evidence.

**Plans:** 1 plan

Plans:

- [x] 38-01: Add CK GEMM supported example, metadata/source tests, native
  staging coverage, RDNA 4 E2E registration, and support docs.

### Phase 39: rocWMMA Supported Replacement

**Status:** Complete 2026-05-22

**Goal:** Promote rocWMMA to supported status for RDNA 4 matrix-core GEMM-style
workloads.

**Requirements:** WMM-01, WMM-02, WMM-03, WMM-04

**Success criteria:**

1. A public `examples/rocwmma/...` problem runs through `sol-execbench` on RDNA 4.
2. The solution source uses real rocWMMA headers/API patterns for the measured implementation.
3. Tests cover rocWMMA metadata, source consistency, dependency diagnostics, and RDNA 4 E2E behavior where hardware is present.
4. Docs identify supported RDNA 4 targets and keep CDNA validation deferred.

**Implementation notes:**

- Treat rocWMMA architecture support as an explicit public contract.
- Prefer a small matrix-core GEMM-style example.
- Do not claim CDNA validation in v1.8.

**Plans:** 1 plan

Plans:

- [x] 39-01: Add rocWMMA matrix-core GEMM supported example,
  metadata/source tests, native staging coverage, RDNA 4 E2E registration, and
  support docs.

### Phase 40: Compatibility Cleanup and RDNA 4 Validation Closure

**Status:** Complete 2026-05-22

**Goal:** Close the library ecosystem gap by removing ambiguous compatibility
claims and recording RDNA 4 validation evidence.

**Requirements:** COMPAT-01, COMPAT-02, COMPAT-03, COMPAT-04, RDNA4-01,
RDNA4-02, RDNA4-03

**Success criteria:**

1. Public docs map former cuDNN, CUTLASS, CuTe DSL, and cuTile categories to supported, retired, or deferred ROCm statuses.
2. Compatibility examples no longer imply supported replacement status unless they contain real runnable library solutions.
3. Public-contract tests enforce support wording and RDNA 4-only validation scope.
4. Focused library example suite passes on RDNA 4 for hipBLAS, MIOpen, CK, and rocWMMA.
5. Closure artifacts summarize supported RDNA 4 library categories and defer CDNA 3/CDNA 4 validation.

**Implementation notes:**

- Keep `hipblas` supported from v1.7 and fold it into final support matrix
  validation.
- Update README, `docs/rocm_libraries.md`, and internal readiness docs together.
- Treat CDNA 3 and CDNA 4 as deferred validation targets.

**Plans:** 1 plan

Plans:

- [x] 40-01: Clean public compatibility/support wording, protect RDNA 4-only
  validation claims, and record focused library validation evidence.

## Completed Phase History

<details>
<summary>v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration (Phases 31-35) - shipped 2026-05-22</summary>

- [x] Phase 31: Optimized Scoring Baseline Semantics (1/1 plan)
- [x] Phase 32: Source-Specific Profiler Timing Workflow (1/1 plan)
- [x] Phase 33: Reward-Hack Defense Expansion (1/1 plan)
- [x] Phase 34: ROCm Library Category Migration (1/1 plan)
- [x] Phase 35: MI300X Validation Readiness Guardrails (1/1 plan)

</details>

## Completed Milestones

- [x] **v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration** - shipped 2026-05-22. See `.planning/milestones/v1.7-ROADMAP.md`.
- [x] **v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow** - shipped 2026-05-22. See `.planning/milestones/v1.6-ROADMAP.md`.
- [x] **v1.5 AMD-native SOL Scoring and ROCm Profiler Timing** - shipped 2026-05-22. See `.planning/milestones/v1.5-ROADMAP.md`.
- [x] **v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness** - shipped 2026-05-22. See `.planning/milestones/v1.4-ROADMAP.md`.
- [x] **v1.3 Non-CDNA Issue Closure** - shipped 2026-05-22. See `.planning/milestones/v1.3-ROADMAP.md`.
- [x] **v1.2 Engineering Practice Harvest and Compatibility Guardrails** - shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.
- [x] **v1.1 CDNA 3 Support and Migration Closure** - shipped 2026-05-21. See `.planning/milestones/v1.1-ROADMAP.md`.
- [x] **v1.0 ROCm Port** - shipped 2026-05-21. See `.planning/milestones/v1.0-ROADMAP.md`.

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
| WMM-01 | Phase 39 | Complete |
| WMM-02 | Phase 39 | Complete |
| WMM-03 | Phase 39 | Complete |
| WMM-04 | Phase 39 | Complete |
| COMPAT-01 | Phase 40 | Complete |
| COMPAT-02 | Phase 40 | Complete |
| COMPAT-03 | Phase 40 | Complete |
| COMPAT-04 | Phase 40 | Complete |
| RDNA4-01 | Phase 40 | Complete |
| RDNA4-02 | Phase 40 | Complete |
| RDNA4-03 | Phase 40 | Complete |
