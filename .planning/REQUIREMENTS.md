# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-06-01
**Active Milestone:** v1.25 Engineering Prerelease
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1.25 Requirements

### Release Candidate Validation

- [ ] **RCVAL-01**: Maintainer can run the CPU-safe release validation suite
  and get a recorded pass/fail summary.
- [ ] **RCVAL-02**: Maintainer can run focused ROCm/Docker smoke checks with
  recorded environment and clock-policy evidence.
- [ ] **RCVAL-03**: Maintainer can run a bounded dataset slice and produce
  trace, closure, trust, and known-gap artifacts.
- [ ] **RCVAL-04**: Release validation failures are classified as blocking,
  deferred, or diagnostic-only with explicit next action.

### Support Matrix

- [ ] **SUPPORT-01**: User can identify which RDNA 4 evidence is validated for
  the prerelease.
- [ ] **SUPPORT-02**: User can distinguish Docker/container user-space
  evidence from native-host validation.
- [ ] **SUPPORT-03**: User can see that MI300X/CDNA3 full-suite validation is
  deferred unless a complete evidence chain exists.
- [ ] **SUPPORT-04**: User can see that CDNA4 validation is unavailable because
  hardware is not currently accessible.

### Claim Boundaries

- [ ] **CLAIM-01**: Release docs prevent paper-parity, upstream SOLAR parity,
  leaderboard, hard-sandbox, and CDNA4 validation overclaims.
- [ ] **CLAIM-02**: Existing claim-boundary tests or docs checks cover the
  prerelease wording.
- [ ] **CLAIM-03**: Release notes clearly state which artifacts are canonical,
  diagnostic-only, provisional, or deferred.

### First-Run User Path

- [ ] **FIRST-01**: New user can install dependencies and run a minimal
  example from documented commands.
- [ ] **FIRST-02**: New user can generate canonical trace JSONL and interpret
  correctness, latency, speedup, and environment fields.
- [ ] **FIRST-03**: New user can diagnose common failures using doctor,
  sidecars, no-trace diagnostics, and known limitations.
- [ ] **FIRST-04**: First-run docs avoid NVIDIA/CUDA ambiguity except where
  PyTorch ROCm compatibility names are intentional.

### Release Materials

- [ ] **REL-01**: Maintainer can follow a prerelease checklist from clean tree
  to tagged release candidate.
- [ ] **REL-02**: Release notes summarize shipped capability, validation
  evidence, known limitations, and deferred claims.
- [ ] **REL-03**: Public docs point users to support matrix, claims,
  researcher guide, timing semantics, and troubleshooting entry points.

## Future Requirements

### Research Validation

- **PAPER-01**: A future milestone can run full 235-problem paper-scale
  validation and compare local AMD-derived outputs with upstream SOLAR where
  applicable.
- **MI300X-01**: A future milestone can run full MI300X/CDNA3 hardware
  validation when a complete evidence chain is available.
- **CDNA4-01**: A future milestone can revisit CDNA4 validation when suitable
  hardware becomes available.

### Operations And Isolation

- **LEADER-01**: A future milestone can design hosted leaderboard submission
  policy, isolation, and operations.
- **SANDBOX-01**: A future milestone can design a hardened OS/container runner
  for adversarial or multi-tenant submissions.

### Dependency And Docker Policy

- **DEP-01**: A future milestone can perform large PyTorch/ROCm relocking or
  Docker privilege redesign if release validation proves it is necessary.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full 235-problem paper-scale validation | Requires paper-scale execution and evidence beyond an engineering prerelease. |
| Upstream SOLAR parity | Requires side-by-side comparison against upstream SOLAR outputs and is not needed for prerelease packaging. |
| MI300X/CDNA3 full-suite validation | Deferred unless a complete real-hardware evidence chain is produced separately. |
| CDNA 3, MI300X, CDNA 4, or native-host ROCm validation expansion | Legacy guardrail wording remains excluded; for v1.25 interpretation, MI300X is the CDNA3 hardware target rather than a separate architecture target. |
| CDNA4 validation | Suitable CDNA4 hardware is not currently available. |
| Hosted leaderboard or remote submission service | Requires service operations, submission policy, and isolation design outside this prerelease. |
| Hard sandbox or multi-tenant adversarial execution | Requires OS/container isolation architecture beyond the current benchmark harness. |
| Large PyTorch/ROCm relock or Docker privilege redesign | Deferred unless prerelease validation exposes a blocking issue. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| RCVAL-01 | Phase 114 | Pending |
| RCVAL-02 | Phase 114 | Pending |
| RCVAL-03 | Phase 114 | Pending |
| RCVAL-04 | Phase 114 | Pending |
| SUPPORT-01 | Phase 115 | Pending |
| SUPPORT-02 | Phase 115 | Pending |
| SUPPORT-03 | Phase 115 | Pending |
| SUPPORT-04 | Phase 115 | Pending |
| CLAIM-01 | Phase 116 | Pending |
| CLAIM-02 | Phase 116 | Pending |
| CLAIM-03 | Phase 116 | Pending |
| FIRST-01 | Phase 117 | Pending |
| FIRST-02 | Phase 117 | Pending |
| FIRST-03 | Phase 117 | Pending |
| FIRST-04 | Phase 117 | Pending |
| REL-01 | Phase 118 | Pending |
| REL-02 | Phase 118 | Pending |
| REL-03 | Phase 118 | Pending |

**Coverage:**
- v1.25 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0

---
*Requirements defined: 2026-06-01*
