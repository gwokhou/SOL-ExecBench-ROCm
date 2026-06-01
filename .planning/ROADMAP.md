# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Complete **v1.25 Engineering Prerelease** - Phases 114-118
  (shipped 2026-06-01). See `.planning/milestones/v1.25-ROADMAP.md`.

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

**Active milestone:** None.

**Status:** v1.25 Engineering Prerelease complete.

**Next step:** Start the next milestone with `$gsd-new-milestone` when ready.

## Recent Outcome

v1.25 packaged the ROCm port for an engineering prerelease by adding bounded
release-candidate validation, a public support matrix, claim-boundary
guardrails, a first-run user path, and release-candidate materials. It did not
add full paper validation, upstream SOLAR parity, hosted leaderboard readiness,
hard sandboxing, MI300X/CDNA3 full-suite validation, CDNA4 validation, or large
dependency/Docker redesign.
