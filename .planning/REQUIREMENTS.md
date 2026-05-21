# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-21
**Milestone:** v1.1 CDNA 3 Support and Migration Closure
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1.1 Requirements

### CDNA 3 Schema and Build Support

- [x] **CDNA-01**: Developer can declare CDNA 3 `gfx94*` solution targets in `solution.json` without schema rejection.
- [x] **CDNA-02**: HIP/C++ packaging injects the correct ROCm offload architecture flags for explicit CDNA 3 targets.
- [x] **CDNA-03**: Local hardware detection and test marker logic consistently treat `gfx94*` as CDNA 3.
- [x] **CDNA-04**: CDNA 3 support is represented as code/schema support, not as hardware-validation evidence.

### Migration Residue Audit

- [ ] **AUDIT-01**: Maintainer can run an audit that reports remaining non-archived CUDA/NVIDIA/CUPTI/library residue.
- [ ] **AUDIT-02**: Each remaining CUDA/NVIDIA term in active source, tests, examples, or docs is either removed, renamed, or allowlisted with a reason.
- [ ] **AUDIT-03**: Compatibility naming from upstream APIs such as PyTorch `torch.cuda` is documented separately from real NVIDIA runtime support.

### Example and Library Closure

- [ ] **EXMP-01**: Former CUTLASS/cuDNN/CuTe/cuTile example categories no longer present ambiguous "fallback" status where benchmark quality requires a stronger ROCm story.
- [ ] **EXMP-02**: Example metadata and tests distinguish ROCm-native implementations, compatibility fallbacks, and intentionally documented non-goals.
- [ ] **EXMP-03**: Example target hardware declarations include CDNA 3 where the implementation is intended to be portable across supported AMD architectures.

### Documentation and Next Validation

- [ ] **DOC-01**: User-facing docs describe supported AMD architecture targets, including CDNA 3 code/schema support and deferred hardware validation.
- [ ] **DOC-02**: Known gaps clearly state that full CDNA 3 suite evidence is planned for the next milestone.
- [ ] **DOC-03**: The next milestone's hardware validation requirements are explicit enough to run and record a `gfx94*` full-suite pass.

## Future Requirements

### Hardware Validation

- **CDNA-HW-01**: Adapted test suite passes on at least one CDNA 3 GPU environment and the full evidence is recorded.
- **CDNA-HW-02**: Documentation can claim CDNA 3 hardware validation after the recorded `gfx94*` full-suite pass.

### Performance Interpretation

- **SCORE-01**: AMD-native scoring or roofline interpretation is defined before making AMD hardware performance claims from SOL-Score-style outputs.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real CDNA 3 hardware validation | User explicitly deferred the actual `gfx94*` full-suite run to the next milestone. |
| Claiming CDNA 3 hardware validation | Code/schema support can be added now, but validation evidence must come from a real CDNA 3 run. |
| Reintroducing CUDA/NVIDIA runtime support | The project remains a ROCm-only port. |
| Perfect one-to-one replacements for every former NVIDIA DSL | This milestone should remove ambiguity and improve important examples, but ROCm-appropriate alternatives remain acceptable. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CDNA-01 | Phase 7 | Complete |
| CDNA-02 | Phase 7 | Complete |
| CDNA-03 | Phase 7 | Complete |
| CDNA-04 | Phase 7 | Complete |
| AUDIT-01 | Phase 8 | Pending |
| AUDIT-02 | Phase 8 | Pending |
| AUDIT-03 | Phase 8 | Pending |
| EXMP-01 | Phase 8 | Pending |
| EXMP-02 | Phase 8 | Pending |
| EXMP-03 | Phase 8 | Pending |
| DOC-01 | Phase 9 | Pending |
| DOC-02 | Phase 9 | Pending |
| DOC-03 | Phase 9 | Pending |

**Coverage:**
- v1.1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0

---
*Requirements defined: 2026-05-21*
*Last updated: 2026-05-21 after v1.1 roadmap creation*
