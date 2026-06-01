# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-06-01
**Active Milestone:** v1.26 Public Prerelease and Research Preview
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1.26 Requirements

### Release Artifact Bundle

- [ ] **ARTIFACT-01**: Maintainer can generate a versioned prerelease artifact
  bundle from a clean checkout or release tag.
- [ ] **ARTIFACT-02**: Maintainer can inspect checksums, command transcripts,
  environment evidence, and validation summaries for the bundle.
- [ ] **ARTIFACT-03**: User can download or locate the bundle and map each
  artifact to its authority class: canonical, diagnostic-only, provisional,
  deferred, or unavailable.

### Release Readiness Gates

- [ ] **GATE-01**: Maintainer can run a release-readiness command or checklist
  that fails when required prerelease artifacts are missing.
- [ ] **GATE-02**: Maintainer can verify that claim boundaries still reject
  paper parity, upstream SOLAR parity, leaderboard readiness, hard-sandbox
  authority, unsupported MI300X/CDNA3 claims, and CDNA4 validation claims.
- [ ] **GATE-03**: Maintainer can see known gaps with explicit blocking,
  deferred, unavailable, or diagnostic-only status before publishing.

### Research Preview Evidence

- [ ] **RESEARCH-01**: Researcher can read a research preview report that
  explains methodology, benchmark scope, hardware scope, evidence surfaces,
  and limitations.
- [ ] **RESEARCH-02**: Researcher can distinguish SOL/AMD-derived evidence from
  upstream SOLAR parity, paper-scale validation, and leaderboard authority.
- [ ] **RESEARCH-03**: Researcher can trace representative commands from
  first-run examples through release validation and bounded dataset-slice
  evidence.

### Public Publishing Materials

- [ ] **PUBLISH-01**: Maintainer can prepare a GitHub prerelease draft or
  equivalent public release page from repository docs.
- [ ] **PUBLISH-02**: Public release materials link the artifact bundle,
  support matrix, claim boundaries, first-run guide, timing semantics,
  researcher guide, and known limitations.
- [ ] **PUBLISH-03**: Public release wording consistently describes the result
  as an engineering prerelease and research preview, not a stable benchmark
  authority release.

## Future Requirements

### Paper-Scale Research Validation

- **PAPER-01**: A future milestone can run complete 235-problem paper-scale
  validation with denominator closure, trace artifacts, failures, and score
  artifacts.
- **SOLAR-01**: A future milestone can compare AMD-derived outputs against
  upstream SOLAR outputs where applicable.

### Hardware Validation

- **MI300X-01**: A future milestone can run full MI300X/CDNA3 validation when a
  complete real-hardware evidence chain is available.
- **CDNA4-01**: A future milestone can revisit CDNA4 validation when suitable
  hardware becomes available.

### Operations And Isolation

- **LEADER-01**: A future milestone can define hosted leaderboard submission,
  scoring, anti-cheat, and hardware policy.
- **SANDBOX-01**: A future milestone can design hardened OS/container isolation
  for adversarial or multi-tenant submissions.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full 235-problem paper-scale validation | Requires paper-scale execution and evidence beyond a prerelease/research-preview package. |
| Upstream SOLAR parity | Requires side-by-side comparison against upstream SOLAR outputs. |
| MI300X/CDNA3 full-suite validation | Deferred unless complete real-hardware evidence becomes available. |
| CDNA4 validation | Suitable CDNA4 hardware is not currently accessible. |
| Hosted leaderboard or remote submission service | Requires service operations, submission policy, and isolation design. |
| Hard sandbox or multi-tenant adversarial execution | Requires OS/container isolation architecture beyond current harness hardening. |
| Stable benchmark authority release | This milestone targets engineering prerelease plus research preview only. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ARTIFACT-01 | Phase 119 | Pending |
| ARTIFACT-02 | Phase 119 | Pending |
| ARTIFACT-03 | Phase 119 | Pending |
| GATE-01 | Phase 120 | Pending |
| GATE-02 | Phase 120 | Pending |
| GATE-03 | Phase 120 | Pending |
| RESEARCH-01 | Phase 121 | Pending |
| RESEARCH-02 | Phase 121 | Pending |
| RESEARCH-03 | Phase 121 | Pending |
| PUBLISH-01 | Phase 122 | Pending |
| PUBLISH-02 | Phase 122 | Pending |
| PUBLISH-03 | Phase 122 | Pending |

**Coverage:**
- v1.26 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0

---
*Requirements defined: 2026-06-01*
