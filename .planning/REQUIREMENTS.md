# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-22
**Milestone:** v1.3 Non-CDNA Issue Closure
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1.3 Requirements

### Original Feature Parity

- [x] **PARITY-01**: Maintainer can review a documented comparison against NVIDIA SOL ExecBench public functionality, including CLI, dataset runner, data download, schemas, trace output, examples, scoring, and supported solution categories.
- [x] **PARITY-02**: Each NVIDIA-specific solution category is classified as ported to ROCm, intentionally replaced, compatibility-example-only, or out of scope, with rationale tied to benchmark semantics.
- [x] **PARITY-03**: Public documentation distinguishes intentional ROCm substitutions from unresolved functional gaps so users do not confuse removed NVIDIA runtime paths with missing ROCm support.

### Scoring and Baseline Comparison

- [x] **SCORE-04**: Maintainer can use a documented AMD-native score or roofline interpretation model before presenting SOL-Score-style values as AMD hardware performance claims.
- [x] **SCORE-05**: User can run a public baseline-comparison CLI or documented workflow over existing trace outputs without changing trace JSONL, solution schema, or benchmark CLI contracts.
- [x] **SCORE-06**: Baseline/scoring outputs include guardrails that identify whether a result is benchmark-relative, baseline-relative, or AMD-native, and block unsupported hardware-performance claims.

### ROCm Library Category Readiness

- [x] **LIB-01**: `hipblas`, `miopen`, `ck`, and `rocwmma` support status is verified against schema validation, native build behavior, example coverage, documentation, and tests.
- [x] **LIB-02**: Any ROCm library category that is not runnable in the current project is explicitly documented as candidate or compatibility-only rather than advertised as fully supported.
- [x] **LIB-03**: Runnable ROCm library examples, if retained as supported categories, use ROCm-facing metadata and have focused tests that protect their public paths and build expectations.

### Engineering Practice Adaptation

- [x] **ENG-01**: Selected `hip-execbench` engineering practices are reviewed for baseline comparison, reporting, validation, and workflow robustness without importing incompatible architecture.
- [x] **ENG-02**: Accepted `hip-execbench` practices are adapted only where they preserve SOL ExecBench public schemas, CLI behavior, trace contracts, and benchmark semantics.
- [x] **ENG-03**: Rejected `hip-execbench` practices are documented when they would duplicate existing behavior, change public contracts, or weaken SOL ExecBench compatibility.

### Non-CDNA Validation Closure

- [x] **VAL-01**: v1.2 discovery-only validation debt is closed with phase-specific validation artifacts, tests, or an explicit documented reason why no additional artifact is required.
- [x] **VAL-02**: Non-CDNA public contract coverage proves CLI help, dataset runner behavior, trace output, schema behavior, example paths, scoring/baseline reporting, and documentation remain stable.
- [x] **VAL-03**: Final milestone audit confirms the only remaining deferred item is real CDNA 3 `gfx94*` hardware validation and related hardware-validation claims.

## Future Requirements

### Hardware Validation

- **CDNA-HW-01**: Adapted test suite passes on at least one CDNA 3 GPU environment and the full evidence is recorded.
- **CDNA-HW-02**: Documentation can claim CDNA 3 hardware validation after the recorded `gfx94*` full-suite pass.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real CDNA 3 `gfx94*` full adapted-suite hardware validation | User explicitly scoped v1.3 to all issues except CDNA 3 validation. |
| CDNA 3 hardware-validation claims | Claims require real `gfx94*` evidence, which is outside this milestone. |
| Restoring CUDA/NVIDIA runtime compatibility | This repository is intentionally a ROCm-only port. |
| Wholesale migration to `hip-execbench` architecture | `hip-execbench` is a reference for engineering practices, not a replacement product architecture. |
| Changing public problem, solution, workload, trace, or existing benchmark CLI contracts without explicit compatibility rationale | v1.3 is a closure and guardrail milestone, not a public-contract rewrite. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PARITY-01 | Phase 14 | Complete |
| PARITY-02 | Phase 14 | Complete |
| PARITY-03 | Phase 14 | Complete |
| SCORE-04 | Phase 15 | Complete |
| SCORE-05 | Phase 15 | Complete |
| SCORE-06 | Phase 15 | Complete |
| LIB-01 | Phase 16 | Complete |
| LIB-02 | Phase 16 | Complete |
| LIB-03 | Phase 16 | Complete |
| ENG-01 | Phase 17 | Complete |
| ENG-02 | Phase 17 | Complete |
| ENG-03 | Phase 17 | Complete |
| VAL-01 | Phase 18 | Complete |
| VAL-02 | Phase 18 | Complete |
| VAL-03 | Phase 18 | Complete |

**Coverage:**
- v1.3 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-05-22*
*Last updated: 2026-05-22 after v1.3 execution*
