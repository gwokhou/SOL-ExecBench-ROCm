# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Active **v1.25 Engineering Prerelease** - Phases 114-118.

- Complete **v1.24 Dataset Batch Run Trustworthiness** - Phases 110-113
  (shipped 2026-06-01). See `.planning/milestones/v1.24-ROADMAP.md`.

- Complete **v1.23 Evaluation Reliability and Security Hardening** -
  Phases 106-109 (shipped 2026-06-01). See
  `.planning/milestones/v1.23-ROADMAP.md`.

- Complete **v1.22 Concern Closure and Execution Boundary Hardening** -
  Phases 100-105 (shipped 2026-06-01). See
  `.planning/milestones/v1.22-ROADMAP.md`.

- Earlier milestones are archived under `.planning/milestones/`.

## Current Position

**Active milestone:** v1.25 Engineering Prerelease.

**Status:** Ready to plan Phase 114.

**Milestone goal:** Turn the v1.24 ROCm port state into an engineering
prerelease / release-candidate package that external users can install,
validate, and interpret without overstated research or hardware claims.

**Explicitly deferred:** full 235-problem paper-scale validation, upstream
SOLAR parity, hosted leaderboard, hard multi-tenant sandboxing, large
PyTorch/ROCm dependency relocking, Docker privilege-model redesign, MI300X/CDNA3
full-suite validation without a complete evidence chain, and CDNA4 validation
because suitable hardware is not currently accessible.

## Phases

- [x] **Phase 114: Release-Candidate Validation** - Maintainers can run and
  interpret bounded prerelease validation across CPU-safe, ROCm/Docker smoke,
  and dataset-slice checks. (completed 2026-06-01)
- [ ] **Phase 115: Support Matrix Boundaries** - Users can understand exactly
  which ROCm hardware and environment evidence the prerelease supports.
- [ ] **Phase 116: Claim Boundary Guardrails** - Release wording and checks
  prevent accidental overclaims about parity, leaderboard readiness, sandboxing,
  and unavailable hardware validation.
- [ ] **Phase 117: First-Run User Path** - New users can install, run a
  minimal example, inspect trace output, and diagnose common failures.
- [ ] **Phase 118: Release Candidate Materials** - Maintainers have the
  checklist, notes, and public documentation entry points needed to tag and
  publish the engineering prerelease.

## Phase Details

### Phase 114: Release-Candidate Validation
**Goal**: Maintainers can validate the engineering prerelease through bounded,
recorded release-candidate checks before publishing.
**Depends on**: Phase 113
**Requirements**: RCVAL-01, RCVAL-02, RCVAL-03, RCVAL-04
**Success Criteria** (what must be TRUE):
  1. Maintainer can run the CPU-safe release validation suite and review a
     recorded pass/fail summary.
  2. Maintainer can run focused ROCm/Docker smoke checks that capture
     environment and clock-policy evidence.
  3. Maintainer can run a bounded dataset slice that produces trace, closure,
     trust, and known-gap artifacts.
  4. Maintainer can classify every release-validation failure as blocking,
     deferred, or diagnostic-only with an explicit next action.
**Plans**:
  - `114-01-PLAN.md` - Add bounded release-candidate validation wrapper,
    docs, and CPU-safe regression coverage.

### Phase 115: Support Matrix Boundaries
**Goal**: Users can interpret prerelease support accurately across RDNA 4,
container user-space, MI300X/CDNA3, and unavailable CDNA4 validation.
**Depends on**: Phase 114
**Requirements**: SUPPORT-01, SUPPORT-02, SUPPORT-03, SUPPORT-04
**Success Criteria** (what must be TRUE):
  1. User can identify which RDNA 4 evidence is validated for the engineering
     prerelease.
  2. User can distinguish Docker/container user-space evidence from native-host
     validation.
  3. User can see that MI300X/CDNA3 full-suite validation is deferred unless a
     complete evidence chain exists.
  4. User can see that CDNA4 validation is unavailable because suitable hardware
     is not currently accessible.
**Plans**: TBD

### Phase 116: Claim Boundary Guardrails
**Goal**: Release claims stay bounded to engineering-prerelease evidence and do
not imply stronger research, service, parity, sandbox, or hardware authority.
**Depends on**: Phase 115
**Requirements**: CLAIM-01, CLAIM-02, CLAIM-03
**Success Criteria** (what must be TRUE):
  1. User reading release docs cannot reasonably infer paper-parity, upstream
     SOLAR parity, leaderboard, hard-sandbox, or CDNA4 validation claims.
  2. Maintainer can run existing or updated claim-boundary checks that cover
     prerelease wording.
  3. User can tell which release artifacts are canonical, diagnostic-only,
     provisional, or deferred.
**Plans**: TBD

### Phase 117: First-Run User Path
**Goal**: New users can complete the documented first-run workflow and
understand the first trace and common failure signals.
**Depends on**: Phase 116
**Requirements**: FIRST-01, FIRST-02, FIRST-03, FIRST-04
**Success Criteria** (what must be TRUE):
  1. New user can install dependencies and run a minimal example from documented
     commands.
  2. New user can generate canonical trace JSONL and interpret correctness,
     latency, speedup, and environment fields.
  3. New user can diagnose common failures using doctor output, sidecars,
     no-trace diagnostics, and known limitations.
  4. New user does not encounter NVIDIA/CUDA ambiguity in first-run docs except
     where PyTorch ROCm compatibility names are intentional.
**Plans**: TBD

### Phase 118: Release Candidate Materials
**Goal**: Maintainers can package, tag, and publish the engineering prerelease
with coherent release materials and public navigation.
**Depends on**: Phase 117
**Requirements**: REL-01, REL-02, REL-03
**Success Criteria** (what must be TRUE):
  1. Maintainer can follow a prerelease checklist from clean tree to tagged
     release candidate.
  2. User can read release notes that summarize shipped capability, validation
     evidence, known limitations, and deferred claims.
  3. Public documentation points users to the support matrix, claim boundaries,
     researcher guide, timing semantics, and troubleshooting entry points.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 114 -> 115 -> 116 -> 117 -> 118.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 114. Release-Candidate Validation | 1/1 | Complete | 2026-06-01 |
| 115. Support Matrix Boundaries | 0/TBD | Not started | - |
| 116. Claim Boundary Guardrails | 0/TBD | Not started | - |
| 117. First-Run User Path | 0/TBD | Not started | - |
| 118. Release Candidate Materials | 0/TBD | Not started | - |

**Coverage:**
- v1.25 requirements mapped: 18/18
- v1.25 requirements complete: 4/18
- Orphaned requirements: 0
- Duplicate requirement mappings: 0
