# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-22
**Milestone:** v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1.4 Requirements

### Compatibility

- [x] **COMPAT-01**: Maintainer can review a source-grounded inventory of public contracts that v1.4 must not change, including CLI behavior, Pydantic schemas, solution format, trace JSONL, and eval-driver semantics.
- [x] **COMPAT-02**: Public contract guardrail tests fail if v1.4 changes existing CLI help, schema behavior, trace JSONL fields, or supported solution metadata unexpectedly.
- [x] **COMPAT-03**: Any adapted `hip-execbench` practice is classified as accepted, rejected, or deferred with rationale tied to SOL ExecBench ROCm compatibility.

### Engineering Adaptation

- [x] **ENG-04**: Maintainer can inspect internal stage diagnostics for parse/package/compile/evaluate/report-style readiness without replacing the existing benchmark execution path.
- [x] **ENG-05**: Maintainer can generate or inspect derived evidence/report data from existing traces and diagnostics without adding fields to trace JSONL.
- [x] **ENG-06**: Adapted reporting or agent-readable summaries explicitly identify themselves as derived artifacts, not canonical benchmark output.

### Validation Readiness

- [x] **VAL-04**: Maintainer can run or inspect CDNA 3 `gfx94*` validation readiness logic that detects environment capability, expected commands, evidence requirements, and blockers without requiring real CDNA 3 hardware.
- [x] **VAL-05**: CDNA 3 readiness outputs and docs clearly state that readiness is not a hardware-validation pass and do not claim CDNA 3 validation until real `gfx94*` full-suite evidence exists.
- [x] **VAL-06**: RDNA 4 validation path has focused unit tests for new readiness/diagnostic/evidence behavior.

### RDNA 4 E2E

- [ ] **RDNA-01**: The implemented v1.4 path passes the relevant unit test suite on RDNA 4.
- [ ] **RDNA-02**: The implemented v1.4 path has recorded RDNA 4 E2E validation evidence using the existing `sol-execbench` benchmark flow.
- [ ] **RDNA-03**: Final validation confirms v1.4 did not regress reference correctness, timing integrity, reward-hack defenses, ROCm-only schema/build/eval behavior, or compatibility guardrails.

## Future Requirements

### CDNA 3 Hardware Validation

- **CDNA-HW-01**: Adapted test suite passes on at least one real CDNA 3 GPU environment and full evidence is recorded.
- **CDNA-HW-02**: Documentation can claim CDNA 3 hardware validation only after recorded `gfx94*` full-suite pass.

### AMD-Native Scoring

- **SCORE-AMD-01**: AMD-native SOL/roofline interpretation can be introduced after a dedicated model and validation plan exist.

### Statistical Baseline Comparison

- **STAT-BASE-01**: Repeated-sample statistical baseline comparison can be exposed after a stable repeated-run data contract exists.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Changing existing `sol-execbench` CLI behavior | User requested no public interface breakage. |
| Changing public Pydantic schemas or solution format | Current schemas are compatibility contracts with SOL ExecBench ROCm users and tests. |
| Adding fields to canonical trace JSONL for v1.4 reports | Derived evidence/report artifacts must remain separate from trace output. |
| Replacing the existing eval driver with `hip-execbench`-style standalone binary execution | Would risk weakening reference correctness, timing integrity, and reward-hack semantics. |
| Claiming real CDNA 3 hardware validation | v1.4 implements readiness only; real `gfx94*` hardware validation remains future scope. |
| Importing the `hip-execbench` TypeScript/Node stack | v1.4 adapts engineering practices, not a second runtime framework. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| COMPAT-01 | Phase 19 | Complete |
| COMPAT-02 | Phase 19 | Complete |
| COMPAT-03 | Phase 19 | Complete |
| ENG-04 | Phase 20 | Complete |
| ENG-05 | Phase 20 | Complete |
| ENG-06 | Phase 20 | Complete |
| VAL-04 | Phase 21 | Complete |
| VAL-05 | Phase 21 | Complete |
| VAL-06 | Phase 21 | Complete |
| RDNA-01 | Phase 22 | Pending |
| RDNA-02 | Phase 22 | Pending |
| RDNA-03 | Phase 22 | Pending |

**Coverage:**
- v1.4 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0

---
*Requirements defined: 2026-05-22*
*Last updated: 2026-05-22 after v1.4 roadmap creation*
