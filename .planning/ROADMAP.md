# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Active **v1.26 Public Prerelease and Research Preview** - Phases 119-122.

- Complete **v1.25 Engineering Prerelease** - Phases 114-118
  (shipped 2026-06-01). See `.planning/milestones/v1.25-ROADMAP.md`.

- Complete **v1.24 Dataset Batch Run Trustworthiness** - Phases 110-113
  (shipped 2026-06-01). See `.planning/milestones/v1.24-ROADMAP.md`.

- Complete **v1.23 Evaluation Reliability and Security Hardening** -
  Phases 106-109 (shipped 2026-06-01). See
  `.planning/milestones/v1.23-ROADMAP.md`.

- Earlier milestones are archived under `.planning/milestones/`.

## Current Position

**Active milestone:** v1.26 Public Prerelease and Research Preview.

**Status:** Phase 121 complete; ready to plan Phase 122.

**Milestone goal:** Produce a publishable engineering prerelease and research
preview package with versioned validation artifacts, release-readiness checks,
and bounded research interpretation that matches the project's actual ROCm
evidence.

**Explicitly deferred:** full 235-problem paper-scale validation, upstream
SOLAR parity, hosted leaderboard, hard multi-tenant sandboxing, full MI300X
validation on the CDNA3 `gfx942` target without complete real-hardware
evidence, CDNA4 validation while hardware is unavailable, and stable
benchmark-authority claims.

## Phases

- [x] **Phase 119: Versioned Prerelease Artifact Bundle** - Maintainers can
  generate and inspect a versioned artifact bundle from a clean checkout or
  release tag.
- [x] **Phase 120: Release Readiness Gates** - Maintainers can run gates that
  block missing artifacts, stale claim boundaries, and unreviewed known gaps.
- [x] **Phase 121: Research Preview Evidence Package** - Researchers can read
  and trace the bounded methodology, evidence surfaces, and non-claims behind
  the prerelease.
- [ ] **Phase 122: Public Publishing Materials** - Maintainers can prepare a
  GitHub prerelease or equivalent public release page with correct links and
  bounded wording.

## Phase Details

### Phase 119: Versioned Prerelease Artifact Bundle
**Goal**: Maintainers can produce a reviewable prerelease artifact bundle with
checksums, transcripts, environment evidence, validation summaries, and
authority classification.
**Depends on**: Phase 118
**Requirements**: ARTIFACT-01, ARTIFACT-02, ARTIFACT-03
**Success Criteria**:
  1. Maintainer can generate a bundle from a clean checkout or release tag.
  2. Bundle includes checksums, command transcripts, release validation output,
     environment evidence, and a manifest.
  3. User can map each artifact to canonical, diagnostic-only, provisional,
     deferred, or unavailable authority classes.
**Plans**: 119-01-PLAN.md
**Status**: Complete

### Phase 120: Release Readiness Gates
**Goal**: Maintainers can run prerelease gates that fail on missing evidence,
claim-boundary regressions, or unreviewed known gaps.
**Depends on**: Phase 119
**Requirements**: GATE-01, GATE-02, GATE-03
**Success Criteria**:
  1. Gate fails when required artifact bundle files or checksums are missing.
  2. Gate verifies forbidden claim boundaries remain visible and unchanged.
  3. Gate reports every known gap as blocking, deferred, unavailable, or
     diagnostic-only before publishing.
**Plans**: 120-01-PLAN.md
**Status**: Complete

### Phase 121: Research Preview Evidence Package
**Goal**: Researchers can interpret the prerelease as a bounded research
preview with explicit methodology, evidence scope, and non-claims.
**Depends on**: Phase 120
**Requirements**: RESEARCH-01, RESEARCH-02, RESEARCH-03
**Success Criteria**:
  1. Research preview report explains methodology, benchmark scope, hardware
     scope, evidence surfaces, and limitations.
  2. Report distinguishes SOL/AMD-derived evidence from upstream SOLAR parity,
     paper-scale validation, and leaderboard authority.
  3. Report links representative first-run, release validation, and bounded
     dataset-slice commands to produced artifacts.
**Plans**: 121-01-PLAN.md
**Status**: Complete

### Phase 122: Public Publishing Materials
**Goal**: Maintainers can prepare a public prerelease page with artifact links,
support boundaries, and research-preview wording.
**Depends on**: Phase 121
**Requirements**: PUBLISH-01, PUBLISH-02, PUBLISH-03
**Success Criteria**:
  1. Maintainer can generate or fill a GitHub prerelease draft from repository
     docs.
  2. Public materials link artifact bundle, support matrix, claims, first-run
     guide, timing semantics, researcher guide, and known limitations.
  3. Public wording consistently says engineering prerelease and research
     preview, not stable benchmark authority.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 119 -> 120 -> 121 -> 122.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 119. Versioned Prerelease Artifact Bundle | 1/1 | Complete | 2026-06-01 |
| 120. Release Readiness Gates | 1/1 | Complete | 2026-06-01 |
| 121. Research Preview Evidence Package | 1/1 | Complete | 2026-06-01 |
| 122. Public Publishing Materials | 0/TBD | Not started | - |

**Coverage:**
- v1.26 requirements mapped: 12/12
- v1.26 requirements complete: 9/12
- Orphaned requirements: 0
- Duplicate requirement mappings: 0
