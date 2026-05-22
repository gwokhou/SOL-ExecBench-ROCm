# Requirements: SOL ExecBench ROCm Port v1.9

**Defined:** 2026-05-22
**Milestone:** v1.9 AMD SOL/SOLAR Bound Modeling Completion
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1.9 Requirements

### Bound Graph And IR

- [ ] **IR-01**: Maintainer can build a structured AMD SOL bound graph from a
  `Definition` and `Workload` without changing canonical benchmark schemas.
- [ ] **IR-02**: Maintainer can inspect each bound graph node's stable ID,
  operation family, source expression, tensor roles, resolved shapes, dtypes,
  confidence, and rationale.
- [ ] **IR-03**: Maintainer can see unsupported or ambiguous reference-code
  constructs preserved as explicit unsupported/inexact graph evidence instead
  of being silently dropped.
- [ ] **IR-04**: Existing callers of `build_amd_sol_bound_artifact()` and
  related scoring imports continue to work through a compatibility facade.

### Operator Modeling

- [ ] **MODEL-01**: Maintainer can generate auditable FLOP formulas and evidence
  for dense matmul, batched matmul, and `@`/`torch.mm`/`torch.matmul` style
  operations.
- [ ] **MODEL-02**: Maintainer can generate conservative FLOP and byte evidence
  for elementwise arithmetic and activation chains with explicit confidence
  labels.
- [ ] **MODEL-03**: Maintainer can generate axis-aware conservative evidence
  for reductions, normalization/RMSNorm/layer-norm-like patterns, and
  softmax/log-softmax-like patterns.
- [ ] **MODEL-04**: Maintainer can distinguish logical views from estimated
  materialized data movement for reshape, transpose, contiguous, broadcast, and
  dtype-conversion-like operations where reference evidence supports it.
- [ ] **MODEL-05**: Maintainer can inspect per-node read bytes, write bytes,
  intermediate or movement bytes, total estimated bytes, formula inputs, and
  rationale.

### Hardware Model Artifacts

- [ ] **HW-01**: Maintainer can load versioned AMD hardware model JSON artifacts
  with architecture, dtype/path, peak compute, memory bandwidth, clock policy or
  assumptions, source, confidence, validation status, and evidence references.
- [ ] **HW-02**: Invalid hardware model artifacts fail with clear errors for
  missing provenance, non-positive numeric values, unknown validation status, or
  architecture mismatches.
- [ ] **HW-03**: RDNA 4 (`gfx1200`) is the only validation target for v1.9
  model artifacts; CDNA 3 / MI300X and CDNA 4 model entries remain unvalidated
  or deferred unless future milestones provide evidence.
- [ ] **HW-04**: Built-in fallback hardware models, if retained, are labeled
  provisional or unvalidated and use the same validation path as external JSON
  artifacts.

### Bound Artifacts And Coverage

- [ ] **BOUND-01**: Maintainer can serialize and load AMD SOL bound artifact v2
  sidecars containing schema version, derived marker, workload identity, graph,
  work estimates, operation bounds, aggregate bound, hardware model reference,
  warnings, and coverage summary.
- [ ] **BOUND-02**: Maintainer can inspect per-operation compute bound, memory
  bound, limiting resource, confidence, and rationale.
- [ ] **BOUND-03**: Coverage summaries report supported, inexact, and
  unsupported operation counts by family and expose worst confidence for the
  artifact.
- [ ] **BOUND-04**: Aggregate bounds never treat unsupported or missing evidence
  as hidden zero-cost work without warnings or unscored degradation.

### Score And Dataset Integration

- [ ] **SCORE-01**: AMD-native score reports consume v2 bound artifacts and
  preserve trace, timing, SOL-bound, baseline, and hardware-model evidence
  references.
- [ ] **SCORE-02**: Unsupported, inexact, provisional hardware, missing bound,
  and `reference_latency` fallback states propagate deterministic warnings or
  unscored states into workload and suite reports.
- [ ] **SCORE-03**: Dataset runs can optionally emit or reference AMD SOL bound
  sidecars when producing `--amd-score-report`, without changing primary trace
  JSONL output.
- [ ] **SCORE-04**: Suite reports expose scored/unscored counts and
  baseline-source or evidence-summary information sufficient to avoid release
  score misuse.

### Public Contracts And Documentation

- [ ] **DOC-01**: Canonical trace JSONL, primary `sol-execbench` CLI behavior,
  and public definition/workload/solution schemas remain unchanged by v1.9
  bound modeling.
- [ ] **DOC-02**: Documentation explains AMD SOL bound artifact v2 semantics,
  hardware model provenance, confidence labels, unsupported/inexact degradation,
  and RDNA 4-only validation scope.
- [ ] **DOC-03**: Documentation and guardrail tests prevent NVIDIA B200,
  upstream SOLAR, leaderboard-equivalence, CDNA 3 / MI300X validation, and CDNA
  4 validation claims from appearing in v1.9 outputs or docs.

### Validation

- [ ] **VAL-01**: Golden tests cover IR extraction and bound artifacts for
  matmul, batched matmul, elementwise chains, activation, reduction,
  normalization, softmax, data movement, dtype conversion, tuple outputs, and
  unsupported operations.
- [ ] **VAL-02**: Score integration tests cover complete, inexact, unsupported,
  missing baseline, reference-latency fallback, provisional hardware, and
  failed-trace cases.
- [ ] **VAL-03**: Public-contract tests prove canonical traces and primary CLI
  output are unchanged by bound artifact and score generation.
- [ ] **VAL-04**: RDNA 4 validation evidence records focused unit tests and a
  small derived-report/sample run that emits trace JSONL, bound artifacts, and
  AMD-native score output.

## Future Requirements

### Hardware Validation

- **FUT-HW-01**: Run the full adapted suite on AMD CDNA 3 / MI300X hardware and
  record environment, clock, timing, FP8, and artifact evidence before claiming
  CDNA 3 validation.
- **FUT-HW-02**: Validate CDNA 4 once suitable hardware, ROCm support, and
  methodology exist.
- **FUT-HW-03**: Define and validate any NVFP4/MXFP4-like AMD path only when
  suitable hardware and methodology are available.

### Paper-Scale Parity

- **FUT-PAPER-01**: Recreate or adapt the original paper's 124-model
  extraction and curation pipeline after the bound-modeling contract is stable.
- **FUT-PAPER-02**: Run paper-scale coverage analysis across the full downloaded
  SOL ExecBench dataset and prioritize unsupported operator families.
- **FUT-SOLAR-01**: Add additional SOLAR-like lowering or lookup techniques only
  after v1.9's local IR and artifact contracts are stable.

## Out of Scope

| Feature | Reason |
| --- | --- |
| CDNA 3 / MI300X real-hardware validation | Explicitly deferred so v1.9 can focus on modeling correctness and RDNA 4 validation. |
| CDNA 4 validation | No current validation target or evidence path in this milestone. |
| NVFP4/MXFP4 hardware validation | No suitable AMD hardware/methodology path is established for v1.9. |
| NVIDIA B200 or leaderboard equivalence | This ROCm fork produces AMD-derived artifacts and must not claim upstream hardware equivalence. |
| Mutating canonical trace JSONL | Bound and score evidence belongs in derived sidecar artifacts and reports. |
| Running bound modeling inside the eval subprocess | Evaluation isolation should remain focused on correctness and timing. |
| Full original 124-model extraction pipeline | Deferred unless used only as background reference for modeling decisions. |
| New graph/symbolic math framework dependency | Existing stdlib, dataclass, Pydantic, and pytest patterns are sufficient for v1.9. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
| --- | --- | --- |
| IR-01 | TBD | Pending |
| IR-02 | TBD | Pending |
| IR-03 | TBD | Pending |
| IR-04 | TBD | Pending |
| MODEL-01 | TBD | Pending |
| MODEL-02 | TBD | Pending |
| MODEL-03 | TBD | Pending |
| MODEL-04 | TBD | Pending |
| MODEL-05 | TBD | Pending |
| HW-01 | TBD | Pending |
| HW-02 | TBD | Pending |
| HW-03 | TBD | Pending |
| HW-04 | TBD | Pending |
| BOUND-01 | TBD | Pending |
| BOUND-02 | TBD | Pending |
| BOUND-03 | TBD | Pending |
| BOUND-04 | TBD | Pending |
| SCORE-01 | TBD | Pending |
| SCORE-02 | TBD | Pending |
| SCORE-03 | TBD | Pending |
| SCORE-04 | TBD | Pending |
| DOC-01 | TBD | Pending |
| DOC-02 | TBD | Pending |
| DOC-03 | TBD | Pending |
| VAL-01 | TBD | Pending |
| VAL-02 | TBD | Pending |
| VAL-03 | TBD | Pending |
| VAL-04 | TBD | Pending |

**Coverage:**
- v1.9 requirements: 28 total
- Mapped to phases: 0
- Unmapped: 28

---
*Requirements defined: 2026-05-22*
*Last updated: 2026-05-22 after v1.9 requirement definition*
