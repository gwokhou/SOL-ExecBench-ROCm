# Roadmap: SOL ExecBench ROCm Port

## Current Milestone: v1.27 Copyright Provenance Cleanup

**Goal:** Complete release-facing copyright and provenance cleanup so the
Apache-2.0 ROCm port accurately preserves upstream NVIDIA notices only where
they apply and clearly attributes independent ROCm work to this project.

**Scope boundary:** This milestone is release hygiene and provenance cleanup.
It does not rewrite git history, change the project license, perform a full
legal audit, optimize benchmark behavior, add GPU validation, or change
benchmark semantics.

## Phases

### Phase 123: Provenance Classification Policy

**Goal:** Establish a defensible provenance classification policy and identify
which active files may retain NVIDIA copyright notices before bulk edits.

**Requirements:** PROV-01, PROV-02, PROV-03

**Success criteria:**
1. Active source, tests, scripts, docs, examples, Docker files, and release
   materials can be classified as upstream retained, derivative modified,
   independent ROCm work, or generated/planning material.
2. The policy documents how each class maps to SPDX/copyright header handling.
3. A reviewable classification artifact identifies active files or directory
   classes allowed to retain NVIDIA notices.
4. The policy explicitly separates upstream code provenance from paper/method
   citation.

### Phase 124: SPDX Header Cleanup

**Goal:** Apply provenance-based SPDX/copyright cleanup across active files
without removing required upstream notices or misattributing independent ROCm
work.

**Requirements:** COPY-01, COPY-02, COPY-03, COPY-04

**Success criteria:**
1. Upstream-retained files keep applicable NVIDIA and Apache-2.0 notices.
2. Derivative modified files keep applicable NVIDIA notices and add project
   attribution where appropriate.
3. Independent ROCm files no longer carry misleading NVIDIA-only file
   copyright attribution.
4. Generated, planning, and documentation files follow the policy without
   unnecessary source-style blanket headers.

### Phase 125: Compliance And Attribution Documentation

**Goal:** Align public compliance, attribution, release, and research-preview
documentation with the provenance policy.

**Requirements:** COMP-01, COMP-02, COMP-03

**Success criteria:**
1. Compliance documentation explains the Apache-2.0 fork relationship, retained
   upstream notices, and project-owned ROCm contributions.
2. Public docs distinguish paper/method citation from file-level source
   copyright ownership.
3. README, research preview, and release materials do not imply NVIDIA or AMD
   endorsement.
4. Documentation explains that prior blanket headers are corrected through
   ordinary commits rather than git history rewriting.

### Phase 126: Provenance Guardrails And Release Gates

**Goal:** Convert copyright/provenance cleanup into repeatable guardrails that
prevent future blanket header drift.

**Requirements:** GATE-01, GATE-02, GATE-03

**Success criteria:**
1. NVIDIA/CUDA residue audit permits NVIDIA notices only for files classified
   as upstream retained or derivative modified.
2. Prerelease readiness checks fail when provenance docs, header policy, or
   allowed NVIDIA notice classifications are missing or inconsistent.
3. Tests protect against both incorrect NVIDIA-only headers on independent
   ROCm files and removal of required NVIDIA notices from upstream-derived
   files.
4. The public prerelease artifact path can report provenance gate status
   without changing benchmark execution semantics.

## Traceability

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

**Coverage:** 13/13 v1.27 requirements mapped.

## Completed Milestones

- Complete **v1.26 Public Prerelease and Research Preview** - Phases 119-122
  (shipped 2026-06-02). See `.planning/milestones/v1.26-ROADMAP.md`.
- Complete **v1.25 Engineering Prerelease** - Phases 114-118
  (shipped 2026-06-01). See `.planning/milestones/v1.25-ROADMAP.md`.
- Complete **v1.24 Dataset Batch Run Trustworthiness** - Phases 110-113
  (shipped 2026-06-01). See `.planning/milestones/v1.24-ROADMAP.md`.
- Complete **v1.23 Evaluation Reliability and Security Hardening** -
  Phases 106-109 (shipped 2026-06-01). See
  `.planning/milestones/v1.23-ROADMAP.md`.
- Earlier milestones are archived under `.planning/milestones/`.

## Current Position

**Next up:** Phase 123 Provenance Classification Policy.

Start with `$gsd-discuss-phase 123` or `$gsd-plan-phase 123`.
