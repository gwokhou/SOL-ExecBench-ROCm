# Roadmap: SOL ExecBench ROCm Port

## Milestones

- 🔄 **v1.9 AMD SOL/SOLAR Bound Modeling Completion** — Phases 41-46
  (active).

- ✅ **v1.8 ROCm Library Ecosystem Completion** — Phases 36-40 (shipped
  2026-05-22). See `.planning/milestones/v1.8-ROADMAP.md`.

- ✅ **v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library
  Migration** — Phases 31-35 (shipped 2026-05-22). See
  `.planning/milestones/v1.7-ROADMAP.md`.

- ✅ **v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow** —
  Phases 27-30 (shipped 2026-05-22). See
  `.planning/milestones/v1.6-ROADMAP.md`.

- ✅ **v1.5 AMD-native SOL Scoring and ROCm Profiler Timing** — Phases 23-26
  (shipped 2026-05-22). See `.planning/milestones/v1.5-ROADMAP.md`.

- ✅ **v1.4 hip-execbench Engineering Experience Adaptation + Validation
  Workflow Readiness** — shipped 2026-05-22. See
  `.planning/milestones/v1.4-ROADMAP.md`.

- ✅ **v1.3 Non-CDNA Issue Closure** — shipped 2026-05-22. See
  `.planning/milestones/v1.3-ROADMAP.md`.

- ✅ **v1.2 Engineering Practice Harvest and Compatibility Guardrails** —
  shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.

- ✅ **v1.1 CDNA 3 Support and Migration Closure** — shipped 2026-05-21. See
  `.planning/milestones/v1.1-ROADMAP.md`.

- ✅ **v1.0 ROCm Port** — shipped 2026-05-21. See
  `.planning/milestones/v1.0-ROADMAP.md`.

## Current Position

**Active milestone:** v1.9 AMD SOL/SOLAR Bound Modeling Completion

**Milestone goal:** Make the ROCm port's AMD SOL bound pipeline credible,
structured, auditable, and broad enough to support paper-style SOL scoring on
the current ROCm path, with validation scoped to RDNA 4 only.

## v1.9 Phases

| Phase | Name | Goal | Requirements |
| --- | --- | --- | --- |
| 41 | Bound Model Contract And Hardware Artifacts | Establish the v2 bound artifact contract, hardware model JSON loader, and public-contract guardrails before changing estimators. | HW-01, HW-02, HW-03, HW-04, DOC-01 |
| 42 | 3/3 | Complete    | 2026-05-23 |
| 43 | 4/4 | Complete    | 2026-05-23 |
| 44 | Bound Artifact V2 And Coverage Semantics | Serialize/load v2 SOL bound sidecars and make coverage, confidence, warnings, and aggregate bounds deterministic. | BOUND-01, BOUND-02, BOUND-03, BOUND-04 |
| 45 | AMD Score And Dataset Integration | Wire v2 bound artifacts into AMD-native workload/suite score reports and optional dataset sidecar output. | SCORE-01, SCORE-02, SCORE-03, SCORE-04 |
| 46 | Documentation And RDNA 4 Validation Closure | Complete docs, claim guardrails, golden tests, and RDNA 4 validation evidence for the modeling pipeline. | DOC-02, DOC-03, VAL-01, VAL-02, VAL-03, VAL-04 |

## Phase Details

### Phase 41: Bound Model Contract And Hardware Artifacts

**Goal:** Establish the artifact, hardware-model, and public-contract foundation
for v1.9.

**Requirements:** HW-01, HW-02, HW-03, HW-04, DOC-01

**Plans:** 3 plans

Plans:
**Wave 1**

- [ ] 41-01-PLAN.md — Create strict packaged/external AMD hardware model JSON loaders.

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 41-02-PLAN.md — Wire v2 hardware model defaults into AMD SOL and score compatibility surfaces.

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 41-03-PLAN.md — Lock public contract, CLI, schema, Trace, and claim guardrails.

**Success criteria:**

1. Versioned AMD hardware model JSON artifacts can be loaded and validated with
   clear errors for invalid provenance, values, status, or architecture.

2. RDNA 4 `gfx1200` is the only v1.9 validation target in model artifacts; CDNA
   3 / MI300X and CDNA 4 remain explicitly unvalidated/deferred.

3. Built-in fallback models, if retained, flow through the same validation path
   and remain labeled provisional or unvalidated.

4. Tests prove canonical `Trace` JSONL, public schemas, and primary
   `sol-execbench` output remain unchanged by the new derived artifacts.

### Phase 42: Structured Bound Graph IR

**Goal:** Introduce a stable bound graph/IR that downstream formula and artifact
code can consume without depending on raw AST details.

**Requirements:** IR-01, IR-02, IR-03, IR-04

**Plans:** 3/3 plans complete

Plans:
**Wave 1**

- [x] 42-01-PLAN.md — Create structured AMD bound graph IR contract, taxonomy,
  serialization, and AST fallback evidence.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 42-02-PLAN.md — Implement workload-aware dynamic-trace-first extraction,
  dataflow edges, and explicit unsupported/inexact evidence.

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 42-03-PLAN.md — Wire the new IR through AMD SOL compatibility facades and
  public-contract guardrails.

**Success criteria:**

1. `Definition` and `Workload` inputs produce structured graph nodes with
   stable IDs, op families, source expressions, tensor roles, shapes, dtypes,
   confidence, and rationale.

2. Unsupported or ambiguous source constructs remain visible as unsupported or
   inexact graph evidence.

3. Existing public scoring imports, including `build_amd_sol_bound_artifact()`,
   continue to work through a compatibility facade.

4. Golden parser tests cover aliases, tensor methods, chained expressions,
   tuple outputs, and unsupported constructs.

### Phase 43: Operator FLOP/Byte/Movement Modeling

**Goal:** Add auditable FLOP, byte, and memory-movement estimates for common
SOL ExecBench operator families.

**Requirements:** MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05

**Success criteria:**

1. Matmul, batched matmul, and `@`/`torch.mm`/`torch.matmul` style operations
   produce explicit FLOP formulas and dtype-aware byte evidence.

2. Elementwise arithmetic and activation chains produce conservative formulas
   with confidence labels.

3. Reductions, normalization/RMSNorm/layer-norm-like patterns, and
   softmax/log-softmax-like patterns produce axis-aware conservative evidence.

4. View/data-movement operations distinguish logical view, materialized
   movement, broadcast, contiguous, and dtype-conversion-like evidence where
   detectable.

5. Tests assert read bytes, write bytes, intermediate/movement bytes, total
   bytes, formula inputs, and rationale for representative fixtures.

### Phase 44: Bound Artifact V2 And Coverage Semantics

**Goal:** Turn graph and estimate evidence into stable v2 AMD SOL bound
sidecars with deterministic coverage and warning behavior.

**Requirements:** BOUND-01, BOUND-02, BOUND-03, BOUND-04

**Success criteria:**

1. AMD SOL bound artifact v2 sidecars serialize and load with schema version,
   derived marker, workload identity, graph, estimates, op bounds, aggregate
   bound, hardware model reference, warnings, and coverage summary.

2. Every operation reports compute bound, memory bound, limiting resource,
   confidence, and rationale.

3. Coverage summaries report supported, inexact, unsupported counts by family
   plus worst confidence.

4. Unsupported or missing evidence cannot silently improve aggregate bounds;
   warnings or unscored degradation are generated deterministically.

### Phase 45: AMD Score And Dataset Integration

**Goal:** Make v2 bound artifacts usable by AMD-native scoring and dataset
reports without changing canonical traces.

**Requirements:** SCORE-01, SCORE-02, SCORE-03, SCORE-04

**Success criteria:**

1. AMD-native score reports consume v2 artifacts and preserve trace, timing,
   SOL-bound, baseline, and hardware-model evidence refs.

2. Unsupported, inexact, provisional hardware, missing bound, failed trace, and
   `reference_latency` fallback cases propagate deterministic warnings or
   unscored states.

3. Dataset runs can optionally emit or reference AMD SOL sidecars when
   `--amd-score-report` is requested.

4. Suite reports expose scored/unscored counts and baseline/evidence summaries
   sufficient to avoid release score misuse.

### Phase 46: Documentation And RDNA 4 Validation Closure

**Goal:** Close v1.9 with user-facing docs, golden tests, and RDNA 4-scoped
validation evidence.

**Requirements:** DOC-02, DOC-03, VAL-01, VAL-02, VAL-03, VAL-04

**Success criteria:**

1. Documentation explains AMD SOL bound artifact v2 semantics, hardware model
   provenance, confidence labels, unsupported/inexact degradation, and RDNA
   4-only validation scope.

2. Guardrail tests prevent NVIDIA B200, upstream SOLAR, leaderboard-equivalence,
   CDNA 3 / MI300X validation, and CDNA 4 validation claims.

3. Golden tests cover matmul, batched matmul, elementwise chains, activation,
   reduction, normalization, softmax, data movement, dtype conversion, tuple
   outputs, and unsupported operations.

4. Score integration tests cover complete, inexact, unsupported, missing
   baseline, reference-latency fallback, provisional hardware, and failed-trace
   cases.

5. RDNA 4 validation evidence records focused unit tests and a small
   derived-report/sample run that emits trace JSONL, bound artifacts, and
   AMD-native score output.

## Requirements Coverage

| Requirement | Phase | Status |
| --- | --- | --- |
| IR-01 | Phase 42 | Pending |
| IR-02 | Phase 42 | Pending |
| IR-03 | Phase 42 | Pending |
| IR-04 | Phase 42 | Pending |
| MODEL-01 | Phase 43 | Pending |
| MODEL-02 | Phase 43 | Pending |
| MODEL-03 | Phase 43 | Pending |
| MODEL-04 | Phase 43 | Pending |
| MODEL-05 | Phase 43 | Pending |
| HW-01 | Phase 41 | Pending |
| HW-02 | Phase 41 | Pending |
| HW-03 | Phase 41 | Pending |
| HW-04 | Phase 41 | Pending |
| BOUND-01 | Phase 44 | Pending |
| BOUND-02 | Phase 44 | Pending |
| BOUND-03 | Phase 44 | Pending |
| BOUND-04 | Phase 44 | Pending |
| SCORE-01 | Phase 45 | Pending |
| SCORE-02 | Phase 45 | Pending |
| SCORE-03 | Phase 45 | Pending |
| SCORE-04 | Phase 45 | Pending |
| DOC-01 | Phase 41 | Pending |
| DOC-02 | Phase 46 | Pending |
| DOC-03 | Phase 46 | Pending |
| VAL-01 | Phase 46 | Pending |
| VAL-02 | Phase 46 | Pending |
| VAL-03 | Phase 46 | Pending |
| VAL-04 | Phase 46 | Pending |

**Coverage:**

- v1.9 requirements: 28 total
- Mapped to phases: 28
- Unmapped: 0

## Future Candidate Work

- CDNA 3 / MI300X full adapted-suite validation.
- FP8 behavior and performance validation on MI300X.
- CDNA 3 and CDNA 4 validation for supported ROCm library examples and bound
  model hardware inputs.

- Profiler-backed performance comparison reports for supported ROCm library
  examples.

- Original paper model-to-subgraph extraction and curation pipeline adaptation.
- Broader upstream SOLAR parity beyond v1.9 local IR and artifact contracts.
- NVFP4/MXFP4-like validation if suitable AMD hardware support and methodology
  become available.
