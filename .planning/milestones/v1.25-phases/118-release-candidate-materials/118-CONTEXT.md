---
phase: 118
phase_name: "Release Candidate Materials"
created_at: "2026-06-01"
autonomous: true
requirements: [REL-01, REL-02, REL-03]
---

# Phase 118 Context

## Goal

Finish the v1.25 engineering-prerelease materials so a maintainer can move
from a clean tree to a tagged release candidate, and public readers can find
support boundaries, claims, researcher guidance, timing semantics, and
troubleshooting entry points.

## Inputs

- Phase 114 added release-candidate validation commands and docs.
- Phase 115 added the engineering prerelease support matrix.
- Phase 116 added `docs/v1_25_release_notes.md`.
- Phase 117 clarified `docs/GETTING-STARTED.md`.
- `README.md` already lists most public documentation entry points.

## Locked Decisions

- Release materials are documentation and guardrail tests only.
- No tag, push, package upload, or GitHub release is created by this phase.
- The checklist should make tree cleanliness, focused tests, release validation,
  review of claim boundaries, tagging, and pushing explicit.
- Release notes should remain an engineering-prerelease document, not a
  paper-validation or leaderboard document.

## Risks

- A maintainer may tag without rerunning release-candidate validation.
- Public documentation may point to first-run docs but not support matrix,
  claims, timing semantics, researcher guide, or troubleshooting.
- Checklist wording could imply the phase itself performed a release.

## Deferred

- Actually tagging and publishing v1.25.
- Uploading artifacts to a package index or GitHub release.
- Running live ROCm/Docker/dataset validation in this CPU-safe documentation
  phase.
