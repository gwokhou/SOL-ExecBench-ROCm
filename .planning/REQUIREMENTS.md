# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-06-02
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1.27 Requirements

Requirements for the Copyright Provenance Cleanup milestone. Each maps to one
roadmap phase.

### Provenance Classification

- [ ] **PROV-01**: Maintainer can classify active source, tests, scripts, docs,
  examples, Docker files, and release materials as upstream retained,
  derivative modified, independent ROCm work, or generated/planning material.
- [ ] **PROV-02**: Maintainer can review a documented provenance policy that
  explains how each classification maps to SPDX/copyright headers.
- [ ] **PROV-03**: Maintainer can identify which active files are allowed to
  retain NVIDIA copyright notices before any bulk header cleanup is applied.

### Copyright Cleanup

- [ ] **COPY-01**: Upstream-retained files preserve applicable NVIDIA
  copyright and Apache-2.0 license notices.
- [ ] **COPY-02**: Derivative modified files preserve applicable NVIDIA
  notices and add this project's own attribution where appropriate.
- [ ] **COPY-03**: Independent ROCm work no longer carries misleading
  NVIDIA-only file copyright attribution.
- [ ] **COPY-04**: Generated, planning, and documentation files follow the
  provenance policy without unnecessary source-style blanket headers.

### Compliance Documentation

- [ ] **COMP-01**: Public compliance docs explain the fork relationship to
  NVIDIA SOL-ExecBench, Apache-2.0 obligations, retained upstream notices, and
  project-owned ROCm contributions.
- [ ] **COMP-02**: Public docs distinguish paper/method citation from
  file-level source copyright ownership.
- [ ] **COMP-03**: README, research preview, and release materials avoid
  implying NVIDIA or AMD endorsement.

### Guardrails

- [ ] **GATE-01**: NVIDIA/CUDA residue audit becomes provenance-aware and
  rejects NVIDIA-only headers on independent ROCm files.
- [ ] **GATE-02**: Prerelease readiness checks fail when provenance docs,
  header policy, or allowed NVIDIA notice classifications are missing or
  inconsistent.
- [ ] **GATE-03**: Tests protect against removing required NVIDIA notices from
  upstream-retained or derivative files.

## Future Requirements

Deferred to future milestones.

### Compliance Automation

- **AUTO-01**: Maintainer can run a full REUSE conformance workflow if the
  project later decides to adopt REUSE as a formal release gate.
- **AUTO-02**: Maintainer can generate SPDX SBOM or package-level provenance
  reports for downstream consumers.

### Legal Review

- **LEGL-01**: Project has an external legal review of copyright, dependency,
  trademark, and patent implications before any stable benchmark authority
  release.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Git history rewriting for prior blanket headers | Ordinary metadata correction should remain auditable through normal commits unless legal counsel requires history rewriting. |
| Relicensing away from Apache-2.0 | The upstream project and this port remain Apache-2.0. |
| Full legal audit or full dependency-license audit | This milestone is release hygiene and provenance cleanup, not legal advice or comprehensive license review. |
| Performance optimization or benchmark semantic changes | Copyright provenance cleanup must not alter benchmark behavior. |
| New GPU validation, paper-scale validation, leaderboard work, or hard sandboxing | These remain separate deferred validation, operations, and security milestones. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PROV-01 | Phase 123 | Pending |
| PROV-02 | Phase 123 | Pending |
| PROV-03 | Phase 123 | Pending |
| COPY-01 | Phase 124 | Pending |
| COPY-02 | Phase 124 | Pending |
| COPY-03 | Phase 124 | Pending |
| COPY-04 | Phase 124 | Pending |
| COMP-01 | Phase 125 | Pending |
| COMP-02 | Phase 125 | Pending |
| COMP-03 | Phase 125 | Pending |
| GATE-01 | Phase 126 | Pending |
| GATE-02 | Phase 126 | Pending |
| GATE-03 | Phase 126 | Pending |

**Coverage:**
- v1.27 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0

---
*Requirements defined: 2026-06-02*
*Last updated: 2026-06-02 after v1.27 milestone start*
