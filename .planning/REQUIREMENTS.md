# Requirements: SOL ExecBench ROCm Port v1.10

**Defined:** 2026-05-23  
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.  
**Milestone:** v1.10 Paper-Aligned SOLAR Automatic Derivation

## v1.10 Requirements

### Derivation

- [ ] **DERIVE-01**: The derivation pipeline can recognize structurally visible attention patterns, including Q/K/V projections, QK score computation, scale or mask handling, softmax, PV value aggregation, and output projection, with explicit degradation when axes or mask semantics are incomplete.
- [ ] **DERIVE-02**: The derivation pipeline can conservatively recognize MoE routing, top-k selection, expert projection, token dispatch, and combine patterns, with dynamic routing evidence recorded when static cardinality is incomplete.
- [ ] **DERIVE-03**: The derivation pipeline can recognize convolution patterns for 1D, 2D, and 3D convolution, including grouped or depthwise convolution metadata, stride, padding, dilation, and output spatial dimensions.
- [ ] **DERIVE-04**: The derivation pipeline can conservatively recognize SSM/Mamba-like projection, depthwise convolution, scan or state update, gating, and output projection patterns, with degraded evidence when recurrence semantics are incomplete.
- [ ] **DERIVE-05**: The derivation pipeline can recognize embedding, positional, gather, and rotary-like memory-bound structures with index and output-shape evidence.
- [ ] **DERIVE-06**: The derivation pipeline treats linear projection as a first-class semantic family while reusing GEMM-compatible formulas when dimensions are explicit.
- [ ] **DERIVE-07**: The derivation pipeline emits compound-family grouping, subrole, and provenance metadata without mutating canonical trace JSONL or public benchmark schemas.

### Modeling

- [ ] **MODEL-01**: Each newly promoted family emits a family-specific formula kind, formula text, and formula input map.
- [ ] **MODEL-02**: Each newly promoted family emits dtype-aware read, write, intermediate, movement, and total byte evidence.
- [ ] **MODEL-03**: Formula and byte evidence carries tensor shape, dtype, semantic-axis, and extraction-source provenance.
- [ ] **MODEL-04**: The estimator applies deterministic supported, inexact, and unsupported confidence rules based on metadata completeness and recognized semantics.
- [ ] **MODEL-05**: Family estimates convert into per-operation compute bound, memory bound, limiting resource, and SOL-bound evidence.

### Reporting

- [ ] **REPORT-01**: SOLAR sidecars report family-aware coverage, extraction provenance, missing patterns, unsupported patterns, degraded nodes, and estimated nodes.
- [ ] **REPORT-02**: Aggregate SOLAR evidence remains machine-verifiable through parseable `scored`, `degraded`, and `unscored` states.
- [ ] **REPORT-03**: AMD-native scoring returns `None` for unscored SOLAR evidence and preserves warnings for degraded SOLAR evidence.
- [ ] **REPORT-04**: Derived reports preserve AMD-native-derived claim boundaries and include evidence references for formulas, hardware models, coverage, and score eligibility.

### Validation

- [x] **TEST-01**: Golden derivation fixtures cover attention, MoE, convolution, SSM/Mamba, embedding or positional patterns, and linear projection.
- [x] **TEST-02**: Negative and degradation fixtures cover dynamic, partial, unsupported, taxonomy-only, and missing-metadata cases.
- [ ] **TEST-03**: Sidecar parse and serialize round-trip tests cover every new machine-verifiable derivation evidence field.
- [ ] **TEST-04**: Public contract guardrails prove canonical schemas, trace JSONL, primary CLI behavior, and existing public benchmark semantics remain unchanged.
- [ ] **TEST-05**: Claim guardrails prevent v1.10 artifacts from implying paper benchmark parity, NVIDIA Blackwell or B200 equivalence, hosted leaderboard readiness, or new real-hardware validation.

## Future Requirements

### Paper-Scale Dataset And Validation

- **FUTURE-01**: Recreate or ingest the original paper-scale 124-model and 235-problem extraction pipeline.
- **FUTURE-02**: Validate derived SOLAR bounds on MI300X, CDNA 3, and CDNA 4 real hardware.
- **FUTURE-03**: Validate NVFP4 and MXFP4 modeling against real hardware behavior.
- **FUTURE-04**: Provide a hosted leaderboard or submission service compatible with the upstream benchmark experience.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Original 124-model / 235-problem extraction | v1.10 focuses on automatic SOLAR derivation inside the ROCm fork, not paper-scale dataset construction. |
| MI300X, CDNA 3, CDNA 4, NVFP4, or MXFP4 validation | Real-hardware validation is explicitly deferred from this milestone, including CDNA 3 / MI300X real-hardware validation. |
| Hosted leaderboard or submission service | The milestone targets local derivation artifacts and reports only. |
| NVIDIA Blackwell or B200 equivalence | The ROCm fork must not claim equivalence with the paper's NVIDIA target hardware. |
| New framework dependencies such as ONNX, MLIR, Dynamo, sympy, or networkx | Research recommends extending the existing Python, FX, AST, and local scoring stack. |
| Executing submitted solution code for derivation | SOLAR derivation must use canonical problem/reference inputs and must remain independent of candidate performance. |
| Canonical schema or primary CLI changes | Derived evidence belongs in sidecars and opt-in reports, not canonical benchmark artifacts. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
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
| TEST-01 | Phase 47 | Complete |
| TEST-02 | Phase 47 | Complete |
| TEST-03 | Phase 51 | Pending |
| TEST-04 | Phase 52 | Pending |
| TEST-05 | Phase 52 | Pending |

**Coverage:**
- v1.10 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---
*Requirements defined: 2026-05-23*
*Last updated: 2026-05-23 after roadmap creation*
