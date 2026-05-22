# Roadmap: SOL ExecBench ROCm Port

## Completed Milestones

- [x] **v1.2 Engineering Practice Harvest and Compatibility Guardrails** — shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.
- [x] **v1.1 CDNA 3 Support and Migration Closure** — shipped 2026-05-21. See `.planning/milestones/v1.1-ROADMAP.md`.
- [x] **v1.0 ROCm Port** — shipped 2026-05-21. See `.planning/milestones/v1.0-ROADMAP.md`.

## Current Milestone: v1.3 Non-CDNA Issue Closure

**Goal:** Close all known non-CDNA residual gaps by comparing the ROCm port
against NVIDIA SOL ExecBench public functionality and selectively adapting
engineering practices from `hip-execbench`, while preserving existing public
contracts.

**Scope boundary:** Real CDNA 3 `gfx94*` hardware validation and CDNA 3
hardware-validation claims remain deferred.

## Phase 14: Original Feature Parity Audit ✓

**Goal:** Produce a source-backed comparison between NVIDIA SOL ExecBench public
functionality and the ROCm port, separating intentional substitutions from
unresolved gaps.

**Requirements:** PARITY-01, PARITY-02, PARITY-03

**Success criteria:**

1. A maintained parity document lists original NVIDIA functionality across CLI,
   dataset runner, data download, schemas, traces, examples, SOL-Score, and
   solution categories.
2. Every NVIDIA solution category has an explicit ROCm disposition: ported,
   replaced, compatibility-example-only, or out of scope.
3. User-facing docs distinguish deliberate ROCm-only design decisions from
   remaining work.
4. Tests or documentation checks protect the parity classification from drifting
   silently.

## Phase 15: AMD Scoring and Baseline Workflow ✓

**Goal:** Make scoring and baseline comparison usable on ROCm without implying
unsupported AMD hardware-performance claims.

**Requirements:** SCORE-04, SCORE-05, SCORE-06

**Success criteria:**

1. AMD-native scoring or roofline interpretation is documented with clear
   prerequisites, claim levels, and limitations.
2. A public baseline-comparison CLI or documented workflow consumes existing
   trace artifacts without changing trace JSONL or solution schemas.
3. Baseline/scoring outputs label results as benchmark-relative,
   baseline-relative, or AMD-native and warn or block unsupported claims.
4. Focused tests cover score interpretation, baseline comparison behavior, and
   public contract stability.

## Phase 16: ROCm Library Category Readiness ✓

**Goal:** Verify whether `hipblas`, `miopen`, `ck`, and `rocwmma` are truly
supported solution categories or should be documented as candidates or
compatibility-only examples.

**Requirements:** LIB-01, LIB-02, LIB-03

**Success criteria:**

1. Schema, build, dependency, docs, examples, and tests are audited for each
   ROCm library category.
2. Runnable categories have ROCm-facing examples and focused tests for public
   paths and build expectations.
3. Non-runnable categories are no longer advertised as fully supported and have
   clear replacement or candidate language.
4. Compatibility examples remain valid and do not imply unsupported runtime
   coverage.

## Phase 17: hip-execbench Practice Adaptation ✓

**Goal:** Borrow useful engineering practices from
`~/PyCharmMiscProject/hip-playground/hip-execbench` without replacing SOL
ExecBench's architecture or public contracts.

**Requirements:** ENG-01, ENG-02, ENG-03

**Success criteria:**

1. A practice review identifies candidate ideas for baseline comparison,
   reporting, validation, and workflow robustness.
2. Accepted practices are implemented only where they preserve schemas, CLI
   behavior, traces, and benchmark semantics.
3. Rejected practices are documented with concrete reasons.
4. Added practices have tests or documentation checks proportional to their
   public impact.

## Phase 18: Non-CDNA Validation Closure ✓

**Goal:** Close validation debt and prove that the only remaining deferred
project risk is CDNA 3 real hardware validation.

**Requirements:** VAL-01, VAL-02, VAL-03

**Success criteria:**

1. v1.2 discovery-only validation debt is closed with phase-specific validation
   artifacts, focused tests, or explicit non-applicability rationale.
2. Public contract coverage includes CLI help, dataset runner behavior, trace
   output, schema behavior, example paths, scoring/baseline reporting, and docs.
3. Full non-CDNA adapted verification passes on the available ROCm environment.
4. Final milestone audit records CDNA 3 `gfx94*` real hardware validation as the
   only deferred item.

## Coverage

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

**Coverage summary:**

- v1.3 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0
- Completed phases: 5/5
- Phase numbering: continued from v1.2, starting at Phase 14

---
*Roadmap created: 2026-05-22 for v1.3 Non-CDNA Issue Closure*
