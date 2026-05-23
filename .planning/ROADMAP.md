# Roadmap: SOL ExecBench ROCm Port

## Milestones

- 🚧 **v1.10 Paper-Aligned SOLAR Automatic Derivation** — Phases 47-52
  (active).

- ✅ **v1.9 AMD SOL/SOLAR Bound Modeling Completion** — Phases 41-46
  (shipped 2026-05-23). See `.planning/milestones/v1.9-ROADMAP.md`.

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

**Active milestone:** v1.10 Paper-Aligned SOLAR Automatic Derivation.

**Milestone goal:** Upgrade the AMD SOL/SOLAR derived bound pipeline into a
paper-aligned automatic SOLAR derivation system for the ROCm port, while
preserving canonical trace JSONL, public schemas, primary CLI behavior, and
AMD-native-derived claim boundaries.

**Explicitly deferred:** original-paper 124-model / 235-problem extraction,
MI300X/CDNA 3/CDNA 4 validation, NVFP4/MXFP4 validation, hosted leaderboard,
NVIDIA Blackwell/B200 equivalence, and new framework dependencies.

## Phases

- [ ] **Phase 47: Derivation Contract And Golden Fixture Matrix** - Establish
  the v1.10 derivation contract and fixture coverage before expanding
  extraction behavior.

- [ ] **Phase 48: Extraction Pipeline And Semantic Provenance** - Add the
  shared extraction infrastructure for compound groups, subroles, provenance,
  and deterministic confidence.

- [ ] **Phase 49: High-Confidence Family Modeling** - Promote linear
  projection, convolution, embedding/positional, and explicit attention into
  formula-backed SOLAR families.

- [ ] **Phase 50: Degraded Complex Family Modeling** - Add conservative MoE
  and SSM/Mamba derivation paths with explicit degraded evidence.

- [ ] **Phase 51: Sidecar Coverage And Score Guards** - Make coverage,
  aggregate state, sidecar fields, and AMD-native score eligibility
  machine-verifiable.

- [ ] **Phase 52: Dataset Runner And Public Contract Closure** - Close v1.10
  through dataset-runner integration, documentation, and public claim
  guardrails.

## Phase Details

### Phase 47: Derivation Contract And Golden Fixture Matrix

**Goal**: Users and maintainers have a stable v1.10 SOLAR derivation contract and fixture matrix that define expected family recognition, degradation, and negative behavior before implementation expands.

**Depends on**: Phase 46

**Requirements**: TEST-01, TEST-02

**Success Criteria** (what must be TRUE):

  1. Maintainer can inspect golden derivation fixtures for attention, MoE,
     convolution, SSM/Mamba, embedding or positional patterns, and linear
     projection.

  2. Maintainer can inspect negative and degradation fixtures for dynamic,
     partial, unsupported, taxonomy-only, and missing-metadata cases.

  3. Fixture expectations identify the intended SOLAR state, supported family,
     missing evidence, and degradation rationale for each case.

  4. The fixture matrix preserves v1.10 scope boundaries and does not require
     paper-scale dataset extraction or real hardware validation.

**Plans**: 6 plans

Plans:
**Wave 1**

- [ ] 47-01-PLAN.md — Write the sidecar-only SOLAR derivation contract.
- [ ] 47-02-PLAN.md — Add the fixture loader and loader-only schema tests.

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 47-03-PLAN.md — Add attention and MoE fixture batches.
- [ ] 47-04-PLAN.md — Add convolution and SSM/Mamba fixture batches.
- [ ] 47-05-PLAN.md — Add embedding/positional and linear projection fixture batches.

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 47-06-PLAN.md — Add full matrix tests and public claim-boundary guardrails.

### Phase 48: Extraction Pipeline And Semantic Provenance

**Goal**: The derivation pipeline can produce compound-family grouping, subrole, shape, dtype, axis, source, and confidence evidence without changing canonical benchmark artifacts.

**Depends on**: Phase 47

**Requirements**: DERIVE-07, MODEL-03, MODEL-04

**Success Criteria** (what must be TRUE):

  1. User can derive internal SOLAR evidence that records compound-family
     grouping, subroles, and extraction provenance outside canonical trace
     JSONL and public schemas.

  2. User can inspect tensor shape, dtype, semantic-axis, and extraction-source
     provenance for extracted formula and byte evidence.

  3. User can distinguish supported, inexact, and unsupported derivation states
     through deterministic confidence rules.

  4. Existing primary `sol-execbench` behavior and canonical benchmark schemas
     remain unchanged while sidecar-only evidence expands.

**Plans**: TBD

### Phase 49: High-Confidence Family Modeling

**Goal**: Users can derive formula-backed SOLAR evidence for high-confidence families whose dimensions and memory behavior are visible from reference or workload structure.

**Depends on**: Phase 48

**Requirements**: DERIVE-01, DERIVE-03, DERIVE-05, DERIVE-06, MODEL-01, MODEL-02, MODEL-05

**Success Criteria** (what must be TRUE):

  1. User can derive explicit attention evidence for Q/K/V projections, QK
     score computation, scale or mask handling, softmax, PV aggregation, and
     output projection, with degradation when axes or mask semantics are
     incomplete.

  2. User can derive convolution evidence for 1D, 2D, and 3D convolution,
     including grouped or depthwise metadata, stride, padding, dilation, and
     output spatial dimensions.

  3. User can derive embedding, positional, gather, rotary-like, and linear
     projection evidence with shape, index, semantic role, and GEMM-compatible
     formula reuse where dimensions are explicit.

  4. Newly promoted high-confidence families emit family-specific formula
     kinds, formula text, formula inputs, and dtype-aware read, write,
     intermediate, movement, and total byte evidence.

  5. Family estimates convert into per-operation compute bound, memory bound,
     limiting resource, and SOL-bound evidence.

**Plans**: TBD

### Phase 50: Degraded Complex Family Modeling

**Goal**: Users can derive conservative, explicitly degraded SOLAR evidence for MoE and SSM/Mamba-like structures when static metadata is incomplete.

**Depends on**: Phase 49

**Requirements**: DERIVE-02, DERIVE-04

**Success Criteria** (what must be TRUE):

  1. User can derive MoE evidence for routing, top-k selection, expert
     projection, token dispatch, and combine patterns.

  2. MoE derivation records dynamic routing evidence and degraded confidence
     when static cardinality or expert-selection metadata is incomplete.

  3. User can derive SSM/Mamba-like evidence for projection, depthwise
     convolution, scan or state update, gating, and output projection patterns.

  4. SSM/Mamba derivation records degraded evidence when recurrence or state
     update semantics are incomplete instead of silently producing scored
     evidence.

**Plans**: TBD

### Phase 51: Sidecar Coverage And Score Guards

**Goal**: Users can rely on SOLAR sidecars and AMD-native score reports to separate scored, degraded, and unscored derivation evidence without manual interpretation.

**Depends on**: Phase 50

**Requirements**: REPORT-01, REPORT-02, REPORT-03, TEST-03

**Success Criteria** (what must be TRUE):

  1. User can inspect SOLAR sidecars for family-aware coverage, extraction
     provenance, missing patterns, unsupported patterns, degraded nodes, and
     estimated nodes.

  2. User can parse aggregate SOLAR evidence into machine-verifiable `scored`,
     `degraded`, and `unscored` states.

  3. AMD-native scoring returns `None` for unscored SOLAR evidence and
     preserves warnings for degraded SOLAR evidence.

  4. Sidecar parse and serialize round-trip tests cover every new
     machine-verifiable derivation evidence field.

**Plans**: TBD

### Phase 52: Dataset Runner And Public Contract Closure

**Goal**: Users can run v1.10 derivation through the intended reporting surfaces with documentation and guardrails that preserve public contracts and claim boundaries.

**Depends on**: Phase 51

**Requirements**: REPORT-04, TEST-04, TEST-05

**Success Criteria** (what must be TRUE):

  1. Derived reports preserve AMD-native-derived claim boundaries and include
     evidence references for formulas, hardware models, coverage, and score
     eligibility.

  2. Public contract guardrails prove canonical schemas, trace JSONL, primary
     CLI behavior, and existing public benchmark semantics remain unchanged.

  3. Claim guardrails prevent v1.10 artifacts from implying paper benchmark
     parity, NVIDIA Blackwell or B200 equivalence, hosted leaderboard
     readiness, or new real-hardware validation.

  4. Dataset-runner and documentation surfaces explain how to consume v1.10
     SOLAR sidecars without requiring paper-scale extraction or new hardware
     validation.

**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 47. Derivation Contract And Golden Fixture Matrix | 0/6 | Not started | - |
| 48. Extraction Pipeline And Semantic Provenance | 0/TBD | Not started | - |
| 49. High-Confidence Family Modeling | 0/TBD | Not started | - |
| 50. Degraded Complex Family Modeling | 0/TBD | Not started | - |
| 51. Sidecar Coverage And Score Guards | 0/TBD | Not started | - |
| 52. Dataset Runner And Public Contract Closure | 0/TBD | Not started | - |

## Requirements Coverage

| Requirement | Phase | Status |
| --- | --- | --- |
| DERIVE-01 | Phase 49 | Pending |
| DERIVE-02 | Phase 50 | Pending |
| DERIVE-03 | Phase 49 | Pending |
| DERIVE-04 | Phase 50 | Pending |
| DERIVE-05 | Phase 49 | Pending |
| DERIVE-06 | Phase 49 | Pending |
| DERIVE-07 | Phase 48 | Pending |
| MODEL-01 | Phase 49 | Pending |
| MODEL-02 | Phase 49 | Pending |
| MODEL-03 | Phase 48 | Pending |
| MODEL-04 | Phase 48 | Pending |
| MODEL-05 | Phase 49 | Pending |
| REPORT-01 | Phase 51 | Pending |
| REPORT-02 | Phase 51 | Pending |
| REPORT-03 | Phase 51 | Pending |
| REPORT-04 | Phase 52 | Pending |
| TEST-01 | Phase 47 | Pending |
| TEST-02 | Phase 47 | Pending |
| TEST-03 | Phase 51 | Pending |
| TEST-04 | Phase 52 | Pending |
| TEST-05 | Phase 52 | Pending |

**Coverage:**

- v1.10 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

## Future Candidate Work

- Original paper-scale 124-model / 235-problem extraction and curation.
- MI300X, CDNA 3, and CDNA 4 real-hardware validation.
- NVFP4 and MXFP4 validation if a suitable AMD hardware path exists.
- Hosted leaderboard or submission service.
- NVIDIA Blackwell/B200 comparison methodology, if ever scoped as a separate
  non-ROCm claim analysis effort.
