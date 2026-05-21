# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-22
**Milestone:** v1.2 Engineering Practice Harvest and Compatibility Guardrails
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1.2 Requirements

### Practice Harvest

- [ ] **HARV-01**: Maintainer can review a documented mapping from selected `hip-execbench` engineering patterns to SOL ExecBench ROCm opportunities, including explicit accept/reject rationale.
- [ ] **HARV-02**: Accepted practice adaptations are limited to internal implementation, tests, or documentation and do not require consumers to change existing problem, solution, trace, or CLI usage.
- [ ] **HARV-03**: Rejected `hip-execbench` ideas are recorded when they would change public interfaces, duplicate existing behavior, or conflict with SOL ExecBench benchmark semantics.

### Diagnostics and Reporting

- [ ] **DIAG-01**: Operator-facing ROCm diagnostics provide clearer readiness information for available profiling and hardware-introspection tools without changing existing CLI contracts.
- [ ] **DIAG-02**: Internal error handling or failure reporting distinguishes parse, packaging, compile, runtime, verification, timing, and environment failures with actionable messages.
- [ ] **DIAG-03**: Report or analysis helpers preserve existing trace schemas while making benchmark outcomes easier to inspect during local ROCm evaluation.

### Scoring and Comparison Guardrails

- [ ] **SCORE-01**: Maintainer can compare the current SOL-Score-style implementation against `hip-execbench` scoring and baseline-comparison practices and identify which ideas are safe to adapt.
- [ ] **SCORE-02**: Any scoring or comparison changes preserve current output formats and avoid new AMD hardware performance claims without validated interpretation.
- [ ] **SCORE-03**: Documentation continues to state that AMD-native roofline or performance interpretation must be defined before stronger AMD performance claims are made.

### Public Contract Stability

- [ ] **COMPAT-01**: Tests prove `definition.json`, `workload.jsonl`, and `solution.json` schemas remain backward-compatible for supported ROCm inputs.
- [ ] **COMPAT-02**: Tests prove CLI invocation patterns and trace JSONL output contracts remain stable after internal diagnostic or reporting improvements.
- [ ] **COMPAT-03**: Public examples remain valid at their existing documented paths and formats unless a change is explicitly documented as an internal-only cleanup with compatibility coverage.

### CDNA 3 Deferral

- [ ] **CDNADEF-01**: CDNA 3 real hardware validation remains explicitly deferred in project state, docs, and validation handoff materials.
- [ ] **CDNADEF-02**: No v1.2 artifact claims a full `gfx94*` hardware-validation pass unless real evidence is recorded in a future milestone.

## Future Requirements

### Hardware Validation

- **CDNA-HW-01**: Adapted test suite passes on at least one CDNA 3 GPU environment and the full evidence is recorded.
- **CDNA-HW-02**: Documentation can claim CDNA 3 hardware validation after the recorded `gfx94*` full-suite pass.

### Performance Interpretation

- **SCORE-FUT-01**: AMD-native scoring or roofline interpretation is defined before making AMD hardware performance claims from SOL-Score-style outputs.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Changing public problem, solution, workload, trace, or CLI contracts | User explicitly requested borrowing engineering experience without changing external interfaces or formats. |
| Porting `hip-execbench` wholesale | The sibling project is a source of practices, not a replacement architecture for SOL ExecBench ROCm. |
| Real CDNA 3 hardware validation | User explicitly deferred the actual `gfx94*` full-suite run again for this milestone. |
| Claiming new AMD hardware performance validation | Scoring/comparison work may improve guardrails, but stronger claims require separate validated evidence. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| HARV-01 | Phase 10 | Pending |
| HARV-02 | Phase 10 | Pending |
| HARV-03 | Phase 10 | Pending |
| DIAG-01 | Phase 11 | Pending |
| DIAG-02 | Phase 11 | Pending |
| DIAG-03 | Phase 11 | Pending |
| SCORE-01 | Phase 12 | Pending |
| SCORE-02 | Phase 12 | Pending |
| SCORE-03 | Phase 12 | Pending |
| COMPAT-01 | Phase 13 | Pending |
| COMPAT-02 | Phase 13 | Pending |
| COMPAT-03 | Phase 13 | Pending |
| CDNADEF-01 | Phase 13 | Pending |
| CDNADEF-02 | Phase 13 | Pending |

**Coverage:**
- v1.2 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0

---
*Requirements defined: 2026-05-22*
*Last updated: 2026-05-22 after v1.2 roadmap creation*
